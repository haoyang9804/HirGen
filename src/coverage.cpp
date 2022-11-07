#include <coverage.hpp>

// std::set<OPTYPE> Coverage::optypes = {};
std::set<std::tuple<OPTYPE, OPTYPE>> Coverage::opTypeEdges = {};
std::set<std::tuple<OPTYPE, DATATYPE>> Coverage::optypeDatatypeTuple = {};
std::set<std::tuple<OPTYPE, SHAPE_STR_SQUARE>> Coverage::opTypeShape = {};

// std::set<OPTYPE> Coverage::old_optypes = {};
std::set<std::tuple<OPTYPE, OPTYPE>> Coverage::old_opTypeEdges = {};
std::set<std::tuple<OPTYPE, DATATYPE>> Coverage::old_optypeDatatypeTuple = {};
std::set<std::tuple<OPTYPE, SHAPE_STR_SQUARE>> Coverage::old_opTypeShape = {};

int Coverage::score = 0;

void Coverage::load() {
  std::string line;
  OPTYPE optype = "", preoptype = "";
  DATATYPE datatype = "";
  SHAPE_STR_SQUARE shape = {};
  while (std::getline(File::icsv, line)) {
    for (int i = 0; i < 2; i++) {
      auto [token, rest_line] = split(line, "@");
      line = rest_line;
      if (i == 0) {
        auto [optype, shape] = split(token, ";");
        if (!(optype.empty()) && !(shape.empty()))
          load_opTypeShape(optype, shape);
      } else if (i == 1) {
        auto [optype, preoptype] = split(token, ";");
        if (!(optype.empty()) && !(preoptype.empty()))
          load_opTypeEdges(optype, preoptype);
      }
    }
    auto [optype, datatype] = split(line, ";");
    if (!(optype.empty()) && !(datatype.empty()))
      load_optypeDatatypeTuple(optype, datatype);
  }
}

void Coverage::save() {
  size_t maxindex =
      std::max(std::max(opTypeEdges.size(), opTypeShape.size()), optypeDatatypeTuple.size());
  std::vector<std::tuple<OPTYPE, SHAPE_STR_SQUARE>> opTypeShape_vec(opTypeShape.begin(),
                                                                    opTypeShape.end());
  std::vector<std::tuple<OPTYPE, OPTYPE>> opTypeEdges_vec(opTypeEdges.begin(), opTypeEdges.end());
  std::vector<std::tuple<OPTYPE, DATATYPE>> optypeDatatypeTuple_vec(optypeDatatypeTuple.begin(),
                                                                    optypeDatatypeTuple.end());
  for (int i = 0; i < maxindex; i++) {
    File::ocsv << (i >= opTypeShape_vec.size() ? ""
                                               : std::get<0>(opTypeShape_vec[i]) + ";" +
                                                     std::get<1>(opTypeShape_vec[i]).str()) +
                      "@" +
                      (i >= opTypeEdges_vec.size() ? ""
                                                   : std::get<0>(opTypeEdges_vec[i]) + ";" +
                                                         std::get<1>(opTypeEdges_vec[i])) +
                      "@" +
                      (i >= optypeDatatypeTuple_vec.size()
                           ? ""
                           : std::get<0>(optypeDatatypeTuple_vec[i]) + ";" +
                                 std::get<1>(optypeDatatypeTuple_vec[i]))
               << std::endl;
  }
}

void Coverage::update(const OPTYPE& optype, const DATATYPE& datatype, const SHAPE_STR_SQUARE& shape,
                      const OPTYPE& preoptype) {
  if (!optype.empty() && !preoptype.empty()) update_opTypeEdges(optype, preoptype);
  if (!optype.empty() && !datatype.empty()) update_optypeDatatypeTuple(optype, datatype);
  if (!optype.empty() && !shape.empty()) update_opTypeShape(optype, shape);
}

bool Coverage::noProgress(const OPTYPE& optype, const DATATYPE& datatype, const SHAPE& shape,
                          const OPTYPE& preoptype) {
  bool oldOptypeShapeTuple = !shape.empty()
                                 ? opTypeShape.count(std::make_tuple(optype, shape)) ||
                                       old_opTypeShape.count(std::make_tuple(optype, shape))
                                 : true;
  bool oldEdge = preoptype != "" ? opTypeEdges.count(std::make_tuple(preoptype, optype)) ||
                                       old_opTypeEdges.count(std::make_tuple(preoptype, optype))
                                 : true;
  bool oldOptypeDatatypeTuple = optypeDatatypeTuple.count(std::make_tuple(optype, datatype)) ||
                                old_optypeDatatypeTuple.count(std::make_tuple(optype, datatype));
  int tmp_score = !oldOptypeShapeTuple + !oldEdge + !oldOptypeDatatypeTuple;
  bool noprogress;
  if (tmp_score == 3) noprogress = false;
  else if (tmp_score == 2) noprogress = (rand() % 100) > 50 ? true : false;
  else if (tmp_score == 1) noprogress = (rand() % 100) > 10 ? true : false;
  else noprogress = true;
  if (!noprogress) {
    update_score(!oldOptypeShapeTuple + !oldEdge + !oldOptypeDatatypeTuple);
    logger(optype + " " + datatype + " SUCCESS");
    if (!oldOptypeShapeTuple)
      logger(">>> New optype_shape_tuple: (" + optype + ", " + SHAPE_to_string(shape, "[", "]") +
             ")");
    if (!oldEdge) logger(">>> New edge: (" + preoptype + ", " + optype + ")");
    if (!oldOptypeDatatypeTuple)
      logger(">>> New optype_datatype_tuple: (" + optype + ", " + datatype + ")");
    update(optype, datatype, shape, preoptype);
  } else {
    logger(optype + " " + datatype + " FAIL");
    logger("<<< Old optye_shape_tuple: (" + optype + ", " + SHAPE_to_string(shape, "[", "]") + ")");
    logger("<<< Old edge: (" + preoptype + ", " + optype + ")");
    logger("<<< Old optype_datatype_tuple: (" + optype + ", " + datatype + ")");
  }
  return noprogress;
}

int Coverage::get_score() {
  std::ofstream tmpwrite("score.txt", std::ios_base::out);
  tmpwrite << score;
  tmpwrite.close();
  return score;
}
