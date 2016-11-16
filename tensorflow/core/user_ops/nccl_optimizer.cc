
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

class NcclOptimization : public GraphOptimizationPass {
 public:
  static int count_;
  Status Run(const GraphOptimizationPassOptions& options) {
    VLOG(0) << "NcclOptimization::Run";
    ++count_;

    Graph* graph = options.graph->get();

    for (Node* node : graph->nodes()) {
      VLOG(0) << SummarizeNodeDef(node->def());

      if (node->type_string() == "Identity") {
        for (Edge const* edge : node->out_edges()) {
          if (edge->IsControlEdge()) continue;
          const Node* dst = edge->dst();
          VLOG(0) << "Output to " << dst->def().device();
          // TODO(pbar) Replace with Nccl Broadcast.
        }
      }

      if (node->type_string() == "AddN") {
        for (Edge const* edge : node->in_edges()) {
          if (edge->IsControlEdge()) continue;
          const Node* src = edge->src();
          VLOG(0) << "Input from " << src->def().device();
          // SummarizeNodeDef(src->def());
          // user specified device: src->def().device()
          // runtime assigned device: src->assigned_device_name();
        }
        // TODO(pbar) Replace with Nccl Reduce.
        // graph->AddEdge(src, i, dst, j);
        // graph->RemoveEdge(edge);
      }
    }

    return Status::OK();
  }
};

int NcclOptimization::count_;
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      NcclOptimization);

}  // namespace tensorflow
