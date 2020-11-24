file(REMOVE_RECURSE
  "../../lib/libact_visitor.a"
  "../../lib/libact_visitor.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/act_visitor.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
