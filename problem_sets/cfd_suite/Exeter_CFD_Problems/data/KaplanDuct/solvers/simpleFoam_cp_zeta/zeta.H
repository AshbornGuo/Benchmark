
                scalar sumField4 = 0;
                scalar sumField5 = 0;
                scalar sumField7 = 0;
                scalar sumField8 = 0;
                scalar sumField9 = 0;
                scalar sumField10 = 0;
                scalar sumField11 = 0;
                scalar sumField12 = 0;
                if (area1 > 0)
                {
                    sumField4 = gSum
                    (
                    mesh.magSf().boundaryField()[inletIndex]*p.boundaryField()[inletIndex]
                    )/area1;
                }

               if (area1 > 0)
                {
                    sumField5 = gSum
                    (
                    mesh.magSf().boundaryField()[inletIndex]*(magSqr(Uw2))
                    )/area1;
                }

               if (area1 > 0) //Ptot inlet
                {
                    sumField7 = (1000*sumField4)+(1000*sumField5/2);
                }

                if (area2 > 0)
                {
                    sumField8 = gSum
                    (
                     mesh.magSf().boundaryField()[outletIndex]*p.boundaryField()[outletIndex] //magSqr(u))/(2*9.81))*(mesh.magSf().boundaryField()[outletIndex]*mag(u))
                    )/area2;
                }

              if (area2 > 0)
                {
                    sumField9 = gSum
                    (
                     mesh.magSf().boundaryField()[outletIndex]*(magSqr(Uw))
                    )/area2;
                }

              if (area2 > 0)
                {
                    sumField10 = (1000*sumField8)+(1000*sumField9/2);
                }


               if (area1 > 0)
                {
                    sumField11 = gSum
                    (
                    mesh.magSf().boundaryField()[inletIndex]*(w2)
                    )/area1;
                }

              if (area2 > 0)
                {
                    sumField12 = gSum
                    (
                     mesh.magSf().boundaryField()[outletIndex]*(magSqr(u))
                    )/area2;
                }

    volScalarField zeta
    (
        IOobject
        (
            "zeta",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
            mesh,
        dimensionedScalar("zeta", dimensionSet(0, 0, 0, 0, 0),1)
    ); 

    volScalarField zeta1
    (
        IOobject
        (
            "zeta1",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
            mesh,
        dimensionedScalar("zeta1", dimensionSet(0, 0, 0, 0, 0),1)
    ); 

                volScalarField u_flow = U.component(0);               
                volScalarField v_flow = U.component(1);      
                volScalarField w_flow = U.component(2); 


                scalar Pressure_drop2 = (6*sumField4/(sumField5*3)) - (2*sumField8*sumField12/(sumField5*3)) + (sumField5*3/(sumField5*3)) - (sumField9*sumField12/(sumField5*3));

                volScalarField U_new = u_flow/U_hack;
                volScalarField U_new2 = v_flow/U_hack;
                volScalarField U_new1 = w_flow/U_hack;
                volScalarField U_new3 = mag(U)/U_hack;



zeta = (6*sumField4/(sumField5*3)) - (2*p_new*U_new/(sumField5*3)) + (sumField5*3/(sumField5*3)) - ((U_new*U_new+U_new1*U_new1+U_new2*U_new2)*U_new/(sumField5*3));

//zeta = (6*sumField4/(sumField5*3)) + (2*p_new*U_new1/(sumField5*3)) + (sumField5*3/(sumField5*3)) + ((U_new*U_new+U_new1*U_new1+U_new2*U_new2)*U_new1/(sumField5*3));


zeta1 = (2*sumField4/(sumField5)) - (2*p_new/(sumField5)) + (sumField5/(sumField5)) - ((U_new*U_new+U_new1*U_new1+U_new2*U_new2)/(sumField5));

//zeta1 = (2*sumField4/(sumField5)) - (2*p_new/(sumField5)) + (sumField5/(sumField5)) - ((U_new*U_new+U_new1*U_new1+U_new2*U_new2)/(sumField5));



		//zeta = ((sumField7*sumField11)+(1000*p_new+0.5*1000*(sqr(U_new)+sqr(U_new1)+sqr(U_new2)))*U_new)/(1000*sumField6*sumField11/2); 
                //zeta = (sumField7+(p_new+0.5*sqr(U_new)))/(1000*sumField6/2); //sumField6

               Info << "    Energy loss = " << Pressure_drop2 <<  " between inlet and outlet " << endl;

               using std::ios;
if (Pstream::master()){
               ofstream Pressure_drop_iteration2;
               Pressure_drop_iteration2.open("zeta_iteration_outflow_outflow",ios::app);
               Pressure_drop_iteration2 << t << "," << Pressure_drop2 << std::endl;
               Pressure_drop_iteration2.close();
}
