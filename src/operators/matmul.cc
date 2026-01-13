#include "operators/matmul.h"
#include "core/common.h"
#include "core/runtime.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================

        const auto& lhs = inputs[0];
        const auto& rhs = inputs[1];

        Shape lhs_shape = lhs->getDims();
        Shape rhs_shape = rhs->getDims();
        IT_ASSERT(lhs_shape.size() == rhs_shape.size() &&
                  lhs_shape.size() >= 2);

        const int n = lhs_shape[lhs_shape.size() - (transA ? 1 : 2)];
        const int m = rhs_shape[rhs_shape.size() - (transB ? 2 : 1)];
        IT_ASSERT(lhs_shape[lhs_shape.size() - (transA ? 2 : 1)] ==
                  rhs_shape[rhs_shape.size() - (transB ? 1 : 2)]);

        lhs_shape.pop_back();
        lhs_shape.pop_back();
        rhs_shape.pop_back();
        rhs_shape.pop_back();

        Shape retval = infer_broadcast(lhs_shape, rhs_shape);
        retval.push_back(n);
        retval.push_back(m);

        return {{retval}};
    }

    } // namespace infini
