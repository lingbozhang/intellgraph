add_subdirectory(edge)
add_subdirectory(graph)
add_subdirectory(proto)
add_subdirectory(visitor)
add_subdirectory(solver)
add_subdirectory(utility)

# Installs IntellGraph include headers
install(
  FILES 
    edge.h
    eigen.h
    graph.h 
    visitor.h
  DESTINATION 
    ${INTELLGRAPH_INCLUDE_DIR}/intellgraph
) 
