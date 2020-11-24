file(REMOVE_RECURSE
  "../lib/libintellgraph.a"
  "../lib/libintellgraph.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/intellgraph.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
