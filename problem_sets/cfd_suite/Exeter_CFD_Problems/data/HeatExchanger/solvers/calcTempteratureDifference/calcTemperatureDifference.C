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
    calcTemperatureDifference

Description
    calculates the Temperature Difference between two patches (average pressure at the patches)

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
    IOdictionary calcTemperatureDifferenceDict
    (
        IOobject
        (
            "calcTemperatureDifferenceDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const word inletName =calcTemperatureDifferenceDict.lookup("inlet");
    const word outletName=calcTemperatureDifferenceDict.lookup("outlet");

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

    // Get times list
    instantList Times = runTime.times();

    // set startTime and endTime depending on -time and -latestTime options
#   include "checkTimeOptions.H"

    Foam::instantList timeDirs = Foam::timeSelector::select0(runTime, args);

    forAll(timeDirs, timei)
    {
        runTime.setTime(timeDirs[timei], timei);

        Info<< "Time = " << runTime.timeName() << endl;

#       include "createT.H"
	
                scalar area1 = gSum(mesh.magSf().boundaryField()[inletIndex]);
                scalar area2 = gSum(mesh.magSf().boundaryField()[outletIndex]);
                scalar sumField1 = 0;
                scalar sumField2 = 0;
                if (area1 > 0)
                {
                    sumField1 = gSum
                    (
                        mesh.magSf().boundaryField()[inletIndex]
                      * T.boundaryField()[inletIndex]
                    ) / area1;
                }

                if (area2 > 0)
                {
                    sumField2 = gSum
                    (
                        mesh.magSf().boundaryField()[outletIndex]
                      * T.boundaryField()[outletIndex]
                    ) / area2;
                }

                scalar Temperature_drop = sumField2-sumField1;
 
               Info << "    Temperature drop = " << Temperature_drop <<  " between inlet and outlet " << endl;

    }

    Info << "End\n" << endl;

    return 0;
}


// ************************************************************************* //
