#include <graphNode.hpp>

std::set<NODE_PTR, node_compare> GNode::outputs = {};
std::set<NODE_PTR, node_compare> GNode::inputs = {};
std::unordered_map<int, NODE_PTRs> GNode::layerToNodeptrs = {};

void GNode::clear() {
  OutputNode::clear();
  InputNode::clear();
  Layer::clear();
}
NODE_PTR GNode::tail_tuple_RelayStmt() {
  using Accessory::name2mname;
  if (OutputNode::size() > 1 || rand() % 2) {
    std::string stat = "output = relay.Tuple([";
    std::string stat_copy = "output2 = relay.Tuple([";
    for (NODE_PTR output : outputs) {
      stat += output->name_() + ",";
      if (name2mname.count(output->name_())) {
        stat_copy += name2mname[output->name_()] + ",";
      } else {
        stat_copy += output->name_() + ",";
      }
    }
    stat += "])";
    stat_copy += "])";
    File::ofile << stat << std::endl;
    File::ofile << stat_copy << std::endl;
    NODE_PTRs outputs_vec(outputs.size());
    std::copy(outputs.begin(), outputs.end(), outputs_vec.begin());
    if (Custom::feature == "nodf") OutputNode::clear();
    return std::make_shared<tupleNode>(assignID(), outputs_vec);
  }
  std::string stat = "output = ";
  std::string stat_copy = "output2 = ";
  for (NODE_PTR output : outputs) {
    stat += output->name_();
    if (name2mname.count(output->name_())) {
      stat_copy += name2mname[output->name_()];
    } else {
      stat_copy += output->name_();
    }
  }
  File::ofile << stat << std::endl;
  File::ofile << stat_copy << std::endl;
  return *(outputs.begin());
}

void OutputNode::clear() { outputs.clear(); }
size_t OutputNode::size() { return outputs.size(); }
bool OutputNode::empty() { return size() == 0; }
NODE_PTRs OutputNode::mirror() {
  NODE_PTRs outputsCopy(size());
  std::copy(outputs.begin(), outputs.end(), outputsCopy.begin());
  return outputsCopy;
}
void OutputNode::insert(NODE_PTR nd) { outputs.insert(nd); }
void OutputNode::erase(NODE_PTR nd) { outputs.erase(nd); }
void OutputNode::update(NODE_PTR newtensor, NODE_PTR lhs, NODE_PTR rhs) {
  insert(newtensor);
  if (lhs != nullptr) erase(lhs);
  if (rhs != nullptr) erase(rhs);
}

void InputNode::clear() { inputs.clear(); }
size_t InputNode::size() { return inputs.size(); }
bool InputNode::empty() { return size() == 0; }
NODE_PTRs InputNode::mirror() {
  NODE_PTRs inputsCopy(size());
  std::copy(inputs.begin(), inputs.end(), inputsCopy.begin());
  return inputsCopy;
}
void InputNode::insert(NODE_PTR nd) { inputs.insert(nd); }
void InputNode::erase(NODE_PTR nd) { inputs.erase(nd); }
std::string InputNode::inputString() {
  std::string inputnames_str = "";
  for (NODE_PTR input : inputs) {
    inputnames_str += input->name_() + ",";
  }
  return inputnames_str;
}

void Layer::clear() { layerToNodeptrs.clear(); }

void Layer::fill(NODE_PTR nd) {
  int layer = nd->layer_();
  if (layerToNodeptrs.count(layer)) {
    layerToNodeptrs[layer].push_back(nd);
  } else {
    layerToNodeptrs.insert({layer, {nd}});
  }
}

NODE_PTRs Layer::get(int id) { return layerToNodeptrs[id]; }