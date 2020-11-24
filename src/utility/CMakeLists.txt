cc_library(
  STATIC
  NAME "utility"
  HDRS
    "util.h"
    "random.h"
  SRCS
    "random.cc"
)

# Installs IntellGraph include headers
install(
  FILES 
    util.h
    random.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
) 
