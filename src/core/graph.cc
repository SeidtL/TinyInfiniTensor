#include "core/graph.h"
#include "core/blob.h"
#include "core/op_type.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    vector<Operator>
    GraphObj::eliminate_redundant_transposes(const Operator& op) {
        if (op->getOpType() != OpType::Transpose) {
            return {};
        }

        auto pred = op->getPredecessors();
        if (pred.size() != 1) {
            return {};
        }
        if (pred[0]->getOpType() != OpType::Transpose) {
            return {};
        }

        // ? -> t1 -> t2 -> ?
        auto t2     = as<TransposeObj>(op);
        auto t1     = as<TransposeObj>(pred[0]);
        auto input  = t1->inputs[0];
        auto output = t2->outputs[0];
        auto p2     = t2->getPermute();

        auto apply = [](const Shape& dims, const Shape& perm) -> Shape {
            Shape modified(dims.size());
            for (size_t i = 0; i < dims.size(); ++i) {
                modified[i] = dims[perm[i]];
            }
            return modified;
        };

        // Check t2(t1(x)) == x
        auto dims     = input->getDims();
        auto modified = input->getDims();
        modified      = apply(modified, t1->getPermute());
        modified      = apply(modified, t2->getPermute());
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] != modified[i]) {
                return {};
            }
        }

        // Remove t1 and t2 from graph
        auto group_pred = t1->getPredecessors();
        auto group_succ = t2->getSuccessors();
        for (auto& pred : group_pred) {
            pred->removeSuccessors(t1);
        }

        input->removeTarget(t1);
        for (auto& succ : group_succ) {
            succ->removePredecessors(t2);
            for (auto& succ_inp : succ->inputs) {
                if (succ_inp.get() == output.get()) {
                    succ_inp = input;
                    break;
                }
            }
            input->addTarget(succ);
        }

        return {pred[0], op};
    }

    vector<Operator>
    GraphObj::eliminate_redundant_matmul_transpose(const Operator& op) {
        if (op->getOpType() != OpType::MatMul) {
            return {};
        }

        auto matmul_op = as<MatmulObj>(op);
        // matmul_op is optimized.
        if (matmul_op->getTransA() || matmul_op->getTransB()) {
            return {};
        }

        vector<Operator> to_remove;
        auto impl = [&]<bool is_left>() {
            constexpr auto matmul_input_idx = is_left ? 0 : 1;
            auto tensor = matmul_op->inputs[matmul_input_idx];
            auto source = tensor->getSource();
            if (!source || source->getOpType() != OpType::Transpose) {
                return;
            }

            auto transpose_op = as<TransposeObj>(source);
            auto perm         = transpose_op->getPermute();
            for (size_t i = 0; i < perm.size() - 2; ++i) {
                if (perm[i] != static_cast<int>(i)) {
                    return;
                }
            }

            // No-op?
            if (perm[perm.size() - 1] != static_cast<int>(perm.size() - 2) ||
                perm[perm.size() - 2] != static_cast<int>(perm.size() - 1)) {
                return;
            }

            // connect predecessors of transpose_op to matmul_op.
            auto pred = transpose_op->getPredecessors();
            for (auto& p : pred) {
                p->removeSuccessors(transpose_op);
                matmul_op->addPredecessors(p);
            }
            matmul_op->removePredecessors(transpose_op);

            auto input = transpose_op->inputs[0];
            input->removeTarget(transpose_op);
            input->addTarget(matmul_op);

            if constexpr (is_left) {
                matmul_op->setTransA(true);
            } else {
                matmul_op->setTransB(true);
            }
            matmul_op->inputs[matmul_input_idx] = input;
            to_remove.emplace_back(transpose_op);
        };

        impl.template operator()<true>();
        impl.template operator()<false>();

        return to_remove;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================

        std::unordered_set<Operator> remove_ops;
        std::vector<Operator> to_remove;
        auto executor = [&] (auto &&fn) {
            remove_ops.clear();
            for (auto& op : ops) {
                if (not remove_ops.count(op)) {
                    to_remove = fn(op);
                    remove_ops.insert(to_remove.begin(), to_remove.end());
                }
            }

            for (auto& op : remove_ops) {
                for (auto& output : op->getOutputs()) {
                    removeTensor(output);
                }
                removeOperator(op);
            }
        };

        executor(eliminate_redundant_transposes);
        executor(eliminate_redundant_matmul_transpose);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        std::unordered_map<TensorObj*, size_t> objs;

        for (auto& tensor : tensors) {
            if (objs.count(tensor.get())) {
                continue;
            }

            auto n_bytes       = tensor->getBytes();
            objs[tensor.get()] = allocator.alloc(n_bytes);
        }

        const auto ptr = allocator.getPtr();

        for (auto& [tensorObj, offset] : objs) {
            auto obj_ptr = static_cast<char*>(ptr) + offset;
            auto data    = infini::make_ref<BlobObj>(runtime, obj_ptr);
            tensorObj->setDataBlob(data);
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

    } // namespace infini
