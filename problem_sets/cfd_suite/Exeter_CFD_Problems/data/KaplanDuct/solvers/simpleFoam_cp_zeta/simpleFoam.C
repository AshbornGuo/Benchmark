/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2013 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    simpleFoam

Description
    Steady-state solver for incompressible, turbulent flow

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
//#include "RASModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"
#include "cuttingPlane.H"
#include "sampledCuttingPlane.H"
#include "sampledPlane.H"
#include "cellSet.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "postProcess.H"
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    //simpleControl simple(mesh);

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }
        laminarTransport.correct(); 
        turbulence->correct();
        #include "cp.H"
        #include "zeta.H"

// Plane 1
point pnt(3.605,0,-0.2);
vector direction(1,0,0);
plane pl1(pnt,direction);
// cutting plane triangulated (true), over a mesh subset
cuttingPlane cutPlane1(pl1,mesh,0);
sampledPlane smpPl("smpPl",mesh,pl1);
smpPl.update();
/*
const labelList& cutCells1 = cutPlane1.cutCells();
word setName("someCells");
cellSet currentSet1(mesh, setName, cutCells1);

currentSet1.write();     */

scalarField slicedDesiredField3 = smpPl.sample(Cp);

scalarField slicedDesiredField5 = smpPl.sample(zeta);

scalarField slicedDesiredField7 = smpPl.sample(zeta1);

scalar area_phantom = gSum(smpPl.magSf());

scalar sumField_phantom_Cp = 0;
scalar sumField_phantom_zeta = 0;
scalar sumField_phantom_zeta1 = 0;
 if (area_phantom > 0)
  {
     sumField_phantom_Cp = gSum
        (
         smpPl.magSf()* slicedDesiredField3
        )/ area_phantom;

     sumField_phantom_zeta = gSum
        (
         smpPl.magSf()* slicedDesiredField5
        )/ area_phantom;

     sumField_phantom_zeta1 = gSum
        (
         smpPl.magSf()* slicedDesiredField7
        )/ area_phantom;
   }


if (Pstream::master()){
    ofstream slicedDesiredField2;
    slicedDesiredField2.open("Cp_cost_function",ios::app);
    slicedDesiredField2 << t << "," << sumField_phantom_Cp << std::endl;
    slicedDesiredField2.close();

    ofstream slicedDesiredField6;
    slicedDesiredField6.open("zeta_cost_function",ios::app);
    slicedDesiredField6 << t << "," << sumField_phantom_zeta << std::endl;
    slicedDesiredField6.close();

    ofstream slicedDesiredField8;
    slicedDesiredField8.open("zeta1_cost_function",ios::app);
    slicedDesiredField8 << t << "," << sumField_phantom_zeta1 << std::endl;
    slicedDesiredField8.close();
}

// Create mesh subsetting engine
/*

scalarField slicedDesiredField = cutPlane1.sample(p);
scalar happy = average(slicedDesiredField);

fvMeshSubset subsetter1(mesh);
label patchI = -1;
subsetter1.setLargeCellSubset(currentSet1, patchI, true);


wordList scalarNames(1);
scalarNames[0] = "Cp";
PtrList<volScalarField> scalarFlds1(scalarNames.size());

wordList vectorNames(1);
vectorNames[0] = "U";
PtrList<volVectorField> vectorFlds1(vectorNames.size());

scalarFlds1.set(0, subsetter1.interpolate(Cp));
vectorFlds1.set(0, subsetter1.interpolate(U));

Info<< "Writing subsetted mesh and fields to time " << runTime.value()
<< endl;
subsetter1.subMesh().write();
forAll(scalarFlds1, i)
{
scalarFlds1[i].write();
}
forAll(vectorFlds1, i)
{
vectorFlds1[i].write();
}

Info<< "Reading field p\n" << endl;
volScalarField p
(
IOobject
(
"p",
runTime.timeName(),
mesh,
IOobject::MUST_READ,
IOobject::AUTO_WRITE
),
mesh
);
interpolationCellPoint<scalar> pInterp(p);

plane pl(point(0.0,2.5e-4,0.0), vector(0,1,0));
sampledPlane spl("planeY", mesh, pl);
spl.update();

scalarField pInterpPl(spl.interpolate(pInterp));

SubList<face> splFaces(spl.faces(), spl.faces().size());
primitivePatch primitivePtch(splFaces, spl.points());

primitivePatchInterpolation primPatchInterp(primitivePtch);

scalarField pInterpPlFaces(primPatchInterp.pointToFaceInterpol ate(pInterpPl));
Info << "spl.average(pInterpPlFaces) = " << spl.average(pInterpPlFaces) << endl;

https://www.cfd-online.com/Forums/openfoam-programming-development/82878-averaging-due-parallel-running-cuttingplane.html
https://www.cfd-online.com/Forums/openfoam-post-processing/61321-write-cells-data-intersecting-plane-cuttingplane-3.html
http://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2013/AliAlSam/movingWaveSlides.pdf
https://www.cfd-online.com/Forums/showthread.php?p=202501):
https://www.cfd-online.com/Forums/openfoam-programming-development/98559-extract-velocity-fly.html
https://www.cfd-online.com/Forums/openfoam-programming-development/134928-set-value-all-cells-contained-cellset.html


*/
        runTime.write();
if(runTime.outputTime())
{
Cp.write();
zeta.write();
}
        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
