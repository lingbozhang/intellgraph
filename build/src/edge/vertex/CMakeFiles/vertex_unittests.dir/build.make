# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.18.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.18.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lingbozhang/src/intellgraph2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lingbozhang/src/intellgraph2/build

# Include any dependencies generated for this target.
include src/edge/vertex/CMakeFiles/vertex_unittests.dir/depend.make

# Include the progress variables for this target.
include src/edge/vertex/CMakeFiles/vertex_unittests.dir/progress.make

# Include the compile flags for this target's objects.
include src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make

src/edge/vertex/CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.o: src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make
src/edge/vertex/CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.o: ../src/edge/vertex/input_vertex_impl_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/edge/vertex/CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.o"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.o -c /Users/lingbozhang/src/intellgraph2/src/edge/vertex/input_vertex_impl_test.cc

src/edge/vertex/CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.i"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lingbozhang/src/intellgraph2/src/edge/vertex/input_vertex_impl_test.cc > CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.i

src/edge/vertex/CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.s"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lingbozhang/src/intellgraph2/src/edge/vertex/input_vertex_impl_test.cc -o CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.s

src/edge/vertex/CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.o: src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make
src/edge/vertex/CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.o: ../src/edge/vertex/op_vertex_impl_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/edge/vertex/CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.o"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.o -c /Users/lingbozhang/src/intellgraph2/src/edge/vertex/op_vertex_impl_test.cc

src/edge/vertex/CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.i"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lingbozhang/src/intellgraph2/src/edge/vertex/op_vertex_impl_test.cc > CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.i

src/edge/vertex/CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.s"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lingbozhang/src/intellgraph2/src/edge/vertex/op_vertex_impl_test.cc -o CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.s

src/edge/vertex/CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.o: src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make
src/edge/vertex/CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.o: ../src/edge/vertex/output_vertex_impl_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/edge/vertex/CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.o"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.o -c /Users/lingbozhang/src/intellgraph2/src/edge/vertex/output_vertex_impl_test.cc

src/edge/vertex/CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.i"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lingbozhang/src/intellgraph2/src/edge/vertex/output_vertex_impl_test.cc > CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.i

src/edge/vertex/CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.s"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lingbozhang/src/intellgraph2/src/edge/vertex/output_vertex_impl_test.cc -o CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.s

src/edge/vertex/CMakeFiles/vertex_unittests.dir/relu_test.cc.o: src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make
src/edge/vertex/CMakeFiles/vertex_unittests.dir/relu_test.cc.o: ../src/edge/vertex/relu_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/edge/vertex/CMakeFiles/vertex_unittests.dir/relu_test.cc.o"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vertex_unittests.dir/relu_test.cc.o -c /Users/lingbozhang/src/intellgraph2/src/edge/vertex/relu_test.cc

src/edge/vertex/CMakeFiles/vertex_unittests.dir/relu_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vertex_unittests.dir/relu_test.cc.i"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lingbozhang/src/intellgraph2/src/edge/vertex/relu_test.cc > CMakeFiles/vertex_unittests.dir/relu_test.cc.i

src/edge/vertex/CMakeFiles/vertex_unittests.dir/relu_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vertex_unittests.dir/relu_test.cc.s"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lingbozhang/src/intellgraph2/src/edge/vertex/relu_test.cc -o CMakeFiles/vertex_unittests.dir/relu_test.cc.s

src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.o: src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make
src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.o: ../src/edge/vertex/sigmoid_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.o"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.o -c /Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_test.cc

src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.i"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_test.cc > CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.i

src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.s"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_test.cc -o CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.s

src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.o: src/edge/vertex/CMakeFiles/vertex_unittests.dir/flags.make
src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.o: ../src/edge/vertex/sigmoid_l2_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.o"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.o -c /Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_l2_test.cc

src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.i"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_l2_test.cc > CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.i

src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.s"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lingbozhang/src/intellgraph2/src/edge/vertex/sigmoid_l2_test.cc -o CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.s

# Object files for target vertex_unittests
vertex_unittests_OBJECTS = \
"CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.o" \
"CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.o" \
"CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.o" \
"CMakeFiles/vertex_unittests.dir/relu_test.cc.o" \
"CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.o" \
"CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.o"

# External object files for target vertex_unittests
vertex_unittests_EXTERNAL_OBJECTS =

bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/input_vertex_impl_test.cc.o
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/op_vertex_impl_test.cc.o
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/output_vertex_impl_test.cc.o
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/relu_test.cc.o
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_test.cc.o
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/sigmoid_l2_test.cc.o
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/build.make
bin/vertex_unittests: lib/libgtest.a
bin/vertex_unittests: lib/libgmock.a
bin/vertex_unittests: lib/libgtest_main.a
bin/vertex_unittests: lib/libvertex.a
bin/vertex_unittests: lib/libgtest.a
bin/vertex_unittests: /Users/lingbozhang/.conan/data/glog/0.4.0/_/_/package/908bcd6a8ee05696464b1dd91716bdfcacf54301/lib/libglog.a
bin/vertex_unittests: /Users/lingbozhang/.conan/data/gflags/2.2.2/_/_/package/2ffcb40e4ac22c93d7ea7142dcb7ade34476fea0/lib/libgflags_nothreads.a
bin/vertex_unittests: lib/libproto.a
bin/vertex_unittests: /Users/lingbozhang/.conan/data/protobuf/3.6.1/bincrafters/stable/package/eec6acc43f6348a597c20e5bd28d9e0590a02597/lib/libprotobuf.a
bin/vertex_unittests: src/edge/vertex/CMakeFiles/vertex_unittests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable ../../../bin/vertex_unittests"
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vertex_unittests.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && /usr/local/Cellar/cmake/3.18.2/bin/cmake -D TEST_TARGET=vertex_unittests -D TEST_EXECUTABLE=/Users/lingbozhang/src/intellgraph2/build/bin/vertex_unittests -D TEST_EXECUTOR= -D TEST_WORKING_DIR=/Users/lingbozhang/src/intellgraph2/build/src/edge/vertex -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=vertex_unittests_TESTS -D CTEST_FILE=/Users/lingbozhang/src/intellgraph2/build/src/edge/vertex/vertex_unittests[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=5 -D TEST_XML_OUTPUT_DIR= -P /usr/local/Cellar/cmake/3.18.2/share/cmake/Modules/GoogleTestAddTests.cmake

# Rule to build all files generated by this target.
src/edge/vertex/CMakeFiles/vertex_unittests.dir/build: bin/vertex_unittests

.PHONY : src/edge/vertex/CMakeFiles/vertex_unittests.dir/build

src/edge/vertex/CMakeFiles/vertex_unittests.dir/clean:
	cd /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex && $(CMAKE_COMMAND) -P CMakeFiles/vertex_unittests.dir/cmake_clean.cmake
.PHONY : src/edge/vertex/CMakeFiles/vertex_unittests.dir/clean

src/edge/vertex/CMakeFiles/vertex_unittests.dir/depend:
	cd /Users/lingbozhang/src/intellgraph2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lingbozhang/src/intellgraph2 /Users/lingbozhang/src/intellgraph2/src/edge/vertex /Users/lingbozhang/src/intellgraph2/build /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex /Users/lingbozhang/src/intellgraph2/build/src/edge/vertex/CMakeFiles/vertex_unittests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/edge/vertex/CMakeFiles/vertex_unittests.dir/depend

