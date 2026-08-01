#ifndef ITK_QSIPREP_COMPAT_H
#define ITK_QSIPREP_COMPAT_H

// ITK 5.4/6 convenience macros, back-filled for ITK 5.3.
//
// TORTOISE's main branch targets ITK 6.0b01. This tree is pinned to 5.3rc04 --
// the version the stock qsiprep ANTs image already provides -- so that porting
// upstream fixes costs no change to qsiprep's registration stack. The commits
// being ported use these two macros; both are pure conveniences with no ITK 6
// runtime dependency, so defining them here is sufficient.
//
// Definitions mirror ITK's own (Modules/Core/Common/include/itkMacro.h).

#include "itkMacro.h"

#ifndef itkOverrideGetNameOfClassMacro
#  define itkOverrideGetNameOfClassMacro(thisClass) \
    const char * GetNameOfClass() const override { return #thisClass; }
#endif

#ifndef itkPrintSelfBooleanMacro
#  define itkPrintSelfBooleanMacro(name) \
    os << indent << #name << ": " << (this->m_##name ? "On" : "Off") << std::endl
#endif

#endif  // ITK_QSIPREP_COMPAT_H
