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
include src/node/CMakeFiles/act_visitor.dir/depend.make

# Include the progress variables for this target.
include src/node/CMakeFiles/act_visitor.dir/progress.make

# Include the compile flags for this target's objects.
include src/node/CMakeFiles/act_visitor.dir/flags.make

# Object files for target act_visitor
act_visitor_OBJECTS =

# External object files for target act_visitor
act_visitor_EXTERNAL_OBJECTS =

lib/libact_visitor.a: src/node/CMakeFiles/act_visitor.dir/build.make
lib/libact_visitor.a: src/node/CMakeFiles/act_visitor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lingbozhang/src/intellgraph2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../lib/libact_visitor.a"
	cd /Users/lingbozhang/src/intellgraph2/build/src/node && $(CMAKE_COMMAND) -P CMakeFiles/act_visitor.dir/cmake_clean_target.cmake
	cd /Users/lingbozhang/src/intellgraph2/build/src/node && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/act_visitor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/node/CMakeFiles/act_visitor.dir/build: lib/libact_visitor.a

.PHONY : src/node/CMakeFiles/act_visitor.dir/build

src/node/CMakeFiles/act_visitor.dir/clean:
	cd /Users/lingbozhang/src/intellgraph2/build/src/node && $(CMAKE_COMMAND) -P CMakeFiles/act_visitor.dir/cmake_clean.cmake
.PHONY : src/node/CMakeFiles/act_visitor.dir/clean

src/node/CMakeFiles/act_visitor.dir/depend:
	cd /Users/lingbozhang/src/intellgraph2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lingbozhang/src/intellgraph2 /Users/lingbozhang/src/intellgraph2/src/node /Users/lingbozhang/src/intellgraph2/build /Users/lingbozhang/src/intellgraph2/build/src/node /Users/lingbozhang/src/intellgraph2/build/src/node/CMakeFiles/act_visitor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/node/CMakeFiles/act_visitor.dir/depend

