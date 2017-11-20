#include "GridCoordinate3D.h"
#include "Assert.h"

// TODO: maybe move to header
template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate1DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim1;

template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate2DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim2;

template<class TcoordType, bool doSignChecks>
const Dimension GridCoordinate3DTemplate<TcoordType, doSignChecks>::dimension = Dimension::Dim3;
