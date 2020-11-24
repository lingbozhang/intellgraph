file(REMOVE_RECURSE
  "../lib/libfactory.a"
  "../lib/libfactory.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/factory.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
