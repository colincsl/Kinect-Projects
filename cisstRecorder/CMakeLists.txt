#
# $Id: CMakeLists.txt 3193 2011-11-17 17:11:41Z bvagvol1 $
#
# (C) Copyright 2009-2011 Johns Hopkins University (JHU), All Rights Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

set (REQUIRED_CISST_LIBRARIES cisstCommon
                              cisstVector
                              cisstOSAbstraction
                              cisstMultiTask
                              cisstStereoVision)

# find cisst with all available libraries
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES} QUIET)

if (cisst_FOUND)

  # load cisst configuration
  include (${CISST_USE_FILE})

  find_package (sawOpenNI REQUIRED HINTS "${cisst_saw_BINARY_DIR}/saw/components")
  find_package (OpenCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBS})

  if (sawOpenNI_FOUND)

    include_directories (${sawOpenNI_INCLUDE_DIR})
    link_directories (${sawOpenNI_LIBRARY_DIR})

    # example without cisstStereoVision
    add_executable (oniExKinectTest oniExKinectTest.cpp)
    target_link_libraries (oniExKinectTest sawOpenNI ${OPENNI_LIBRARIES} ${sawOpenNI_LIBRARIES} ${OpenCV_LIBS})
    cisst_target_link_libraries (oniExKinectTest cisstCommon cisstVector cisstOSAbstraction)

    # example with cisstStereoVision
    add_executable (oniExKinectView oniExKinectView.cpp)
    target_link_libraries (oniExKinectView sawOpenNI ${OPENNI_LIBRARIES} ${sawOpenNI_LIBRARIES})
    cisst_target_link_libraries (oniExKinectView ${REQUIRED_CISST_LIBRARIES})



  endif (sawOpenNI_FOUND)

endif (cisst_FOUND)

