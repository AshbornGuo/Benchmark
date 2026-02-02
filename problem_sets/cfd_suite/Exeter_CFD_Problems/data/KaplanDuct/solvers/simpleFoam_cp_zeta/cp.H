#   include "OFstream.H"


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
    const word outletName=calcPressureDifferenceDict.lookup("outlet");
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

                scalar area1 = gSum(mesh.magSf().boundaryField()[inletIndex]); //inlet area
                scalar area2 = gSum(mesh.magSf().boundaryField()[outletIndex]); //outlet area

	vectorField Uw = U.boundaryField()[outletIndex];
    scalarField u = Uw.component(0);
    scalarField v = Uw.component(1);
    scalarField w = Uw.component(2);

	vectorField Uw2 = U.boundaryField()[inletIndex];
    scalarField u2 = Uw2.component(0);
    scalarField v2 = Uw2.component(1);
    scalarField w2 = Uw2.component(2);


                scalar sumField1 = 0;
                scalar sumField2 = 0;
                scalar sumField3 = 0;
                scalar sumField6 = 0;
                if (area1 > 0)
                {
                    sumField1 = gSum
                    (
                        mesh.magSf().boundaryField()[inletIndex]
                      * p.boundaryField()[inletIndex]
                    )/ area1;
                }

                if (area2 > 0)
                {
                    sumField2 = gSum
                    (
                        mesh.magSf().boundaryField()[outletIndex]
                      * p.boundaryField()[outletIndex]
                    )/ area2;
                }


                if (area1 > 0)
                {
                    sumField3 = gSum
                    (
                        mesh.magSf().boundaryField()[inletIndex]
                      * mag(U.boundaryField()[inletIndex])
                    )/ area1;
                }

               if (area1 > 0)
                {
                    sumField6 = gSum
                    (
                    mesh.magSf().boundaryField()[inletIndex]*magSqr(w2)
                    )/area1;
                }
    volScalarField Cp
    (
        IOobject
        (
            "Cp",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
            mesh,
        dimensionedScalar("Cp", dimensionSet(0, 0, 0, 0, 0),1)
    );
                 volScalarField p_new = p/p_hack;
                 scalar Pressure_drop = (sumField2-sumField1)/(0.5*sumField6);
 
                  Cp = (p_new-sumField1)/(0.5*sumField6);

               Info << "    Pressure drop = " << Pressure_drop <<  " between inlet and outlet " << endl;
               scalar t=runTime.value(); 
               using std::ios;
if (Pstream::master()){
               ofstream Pressure_drop_iteration;
               Pressure_drop_iteration.open("Cp_iteration_outflow_outflow",ios::app);
               Pressure_drop_iteration << t << "," << Pressure_drop << std::endl;
               //Pressure_drop_iteration << Pressure_drop << "\n";
               Pressure_drop_iteration.close();
}
