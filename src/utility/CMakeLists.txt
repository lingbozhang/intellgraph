cc_library(
  STATIC
  NAME "utility"
  HDRS
    "random.h"
  SRCS
    "random.cc"
)

# Installs IntellGraph include headers
install(
  FILES 
    ipow.h
    random.h
    util.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
) 
