/*---------------------------------------------------------------------------*\
 =========                   |
 \\      /   F ield          | OpenFOAM: The Open Source CFD Toolbox
  \\    /    O peration      |
   \\  /     A nd            | Copyright (C) 1991-2005 OpenCFD Ltd.
    \\/      M anipulation   |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation; either version 2 of the License, or (at your
    option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM; if not, write to the Free Software Foundation,
    Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

Application
    calcPressureDifference

Description
    calculates the Pressure Difference between two patches (average pressure at the patches)

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#   include "OFstream.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Main program:

int main(int argc, char *argv[])
{
#   include "addTimeOptions.H"

#   include "setRootCase.H"
#   include "createTime.H"
#   include "createMesh.H"

    // Read control dictionary
    IOdictionary calcPressureDifferenceDict
    (
        IOobject
        (
            "calcPressureDifferenceDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const word inletName =calcPressureDifferenceDict.lookup("inlet");
    const word outletName=calcPressureDifferenceDict.lookup("outlet1");
    const word outletName2=calcPressureDifferenceDict.lookup("outlet2");

    label inletIndex=mesh.boundaryMesh().findPatchID(inletName);
    if(inletIndex<0) {
            FatalErrorIn(args.executable())
              << "No patch " << inletName << " in mesh"
              << exit(FatalError);
     }
    label outletIndex=mesh.boundaryMesh().findPatchID(outletName);
    if(outletIndex<0) {
            FatalErrorIn(args.executable())
              << "No patch " << outletName << " in mesh"
              << exit(FatalError);
     }
    label outletIndex2=mesh.boundaryMesh().findPatchID(outletName2);
    if(outletIndex2<0) {
            FatalErrorIn(args.executable())
              << "No patch " << outletName << " in mesh"
              << exit(FatalError);
     }
    // Get times list
    instantList Times = runTime.times();

    // set startTime and endTime depending on -time and -latestTime options
#   include "checkTimeOptions.H"

    /*for (label i=startTime; i<endTime; i++)
    {
        runTime.setTime(Times[i], i);*/

    Foam::instantList timeDirs = Foam::timeSelector::select0(runTime, args);

    forAll(timeDirs, timei)
    {
        runTime.setTime(timeDirs[timei], timei);

        Info<< "Time = " << runTime.timeName() << endl;

#       include "createP.H"
	
                scalar area1 = gSum(mesh.magSf().boundaryField()[inletIndex]);
                scalar area2 = gSum(mesh.magSf().boundaryField()[outletIndex]);
                scalar area3 = gSum(mesh.magSf().boundaryField()[outletIndex2]);
                scalar sumField1 = 0;
                scalar sumField2 = 0;
                scalar sumField3 = 0;
                scalar sumField4 = 0;
                scalar sumField5 = 0;
                scalar sumField6 = 0;

	            vectorField Uo1 = U.boundaryField()[outletIndex];
                scalarField v1 = Uo1.component(1);

	            vectorField Uo2 = U.boundaryField()[outletIndex2];
                scalarField v2 = Uo1.component(1);

                if (area1 > 0)
                {
                    sumField1 = gSum
                    (
                        mesh.magSf().boundaryField()[inletIndex]
                      * p.boundaryField()[inletIndex]
                    ) / area1;
                }

                if (area2 > 0)
                {
                    sumField2 = gSum
                    (
                        mesh.magSf().boundaryField()[outletIndex]
                      * p.boundaryField()[outletIndex]
                    ) / area2;
                }


                if (area3 > 0)
                {
                    sumField3 = gSum
                    (
                        mesh.magSf().boundaryField()[outletIndex2]
                      * p.boundaryField()[outletIndex2]
                    ) / area3;
                }
                if (area1 > 0)
                {
                    sumField4 = gSum
                    (
                        mesh.magSf().boundaryField()[inletIndex]
                      * mag(U.boundaryField()[inletIndex])
                    ) / area1;
                }
                if (area2 > 0)
                {
                    sumField5 = gSum
                    (
                        mesh.magSf().boundaryField()[outletIndex]
                      * v1
                    ) / area2;
                }
                if (area3 > 0)
                {
                    sumField6 = gSum
                    (
                        mesh.magSf().boundaryField()[outletIndex2]
                      * v2
                    ) / area3;
                }
                scalar Pressure_drop = ((sumField1-sumField2)/(0.5*sumField4*sumField4))+((sumField4*sumField4 - sumField5*sumField5)/(sumField4*sumField4));
                scalar Pressure_drop2 = ((sumField1-sumField3)/(0.5*sumField4*sumField4))+((sumField4*sumField4 - sumField6*sumField6)/(sumField4*sumField4));

               Info << "    Total pressure drop = " << Pressure_drop <<  " between Inlet and Outlet1 " << endl;

               Info << "    Total pressure drop = " << Pressure_drop2 <<  " between Inlet and Outlet2 " << endl;
    }

    Info << "End\n" << endl;

    return 0;
}


// ************************************************************************* //
