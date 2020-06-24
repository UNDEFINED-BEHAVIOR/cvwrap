// #include <CL/cl.h>
// #include <CL/cl.hpp>
// #include <CL/cl_platform.h>
// #include <vexcl/vexcl.hpp>

#include <string>
#include "common.h"
#include "cvWrapDeformer.h"
#include "cvWrapCmd.h"

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>


const std::string overrideName = std::string(CVWrap::kName) + std::string("_GPUOverride");

MStatus initializePlugin(MObject obj) { 
  MStatus status;
  MFnPlugin plugin(obj, "Chad Vernon", "1.0", "Any");
  status = plugin.registerNode(CVWrap::kName, CVWrap::id, CVWrap::creator, CVWrap::initialize,
                               MPxNode::kDeformerNode);
  CHECK_MSTATUS_AND_RETURN_IT(status);
  status = plugin.registerCommand(CVWrapCmd::kName, CVWrapCmd::creator, CVWrapCmd::newSyntax);
  CHECK_MSTATUS_AND_RETURN_IT(status);
  status = MGPUDeformerRegistry::registerGPUDeformerCreator(CVWrap::kName, overrideName.c_str(),
  CVWrapGPU::GetGPUDeformerInfo());
  CHECK_MSTATUS_AND_RETURN_IT(status);
  // Set the load path so we can find the cl kernel.
  CVWrapGPU::pluginLoadPath = plugin.loadPath();
  return status;
}

MStatus uninitializePlugin( MObject obj) {
  MStatus status;
  MFnPlugin plugin(obj);

  status = MGPUDeformerRegistry::deregisterGPUDeformerCreator(CVWrap::kName, overrideName.c_str());
  CHECK_MSTATUS_AND_RETURN_IT(status);
  
  status = plugin.deregisterCommand(CVWrapCmd::kName);
  CHECK_MSTATUS_AND_RETURN_IT(status);
  status = plugin.deregisterNode(CVWrap::id);
  CHECK_MSTATUS_AND_RETURN_IT(status);

  return status;
}
