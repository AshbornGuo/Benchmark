*KEYWORD
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                   Control                                   $$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*CONTROL_SHELL
$#  wrpang     esort     irnxx    istupd    theory       bwc     miter      proj
      20.0         1        -1         0         2         1         1         0
$# rotascl    intgrd    lamsht    cstyp6    tshell
       1.0         0         0         1         0
$# psstupd   sidt4tu     cntco    itsflg    irquad 
         0         0         0         0         2
$#  nfail1    nfail4   psnfail    keepcs     delfr   drcpsid    drcprm 
         1         0         0         0         0         0       1.0
*CONTROL_ENERGY
$     hgen      rwen    slnten     rylen
         2         2         1         1
*CONTROL_PARALLEL
$    -----     -----     CONST     -----
         0         0         1         1
*CONTROL_TERMINATION
$   endtim    endcyc     dtmin    endeng    endmas
      45.0         0       0.0       0.0     1.0E8
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                  Database                                   $$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*DATABASE_NODOUT
$#      dt    binary      lcur     ioopt   option1   option2 
      0.05         2         0         1       0.0         0
*DATABASE_RWFORC
$#      dt    binary      lcur     ioopt
      0.05         2         0         1
*DATABASE_RCFORC
$#      dt    binary      lcur     ioopt
      0.05         2         0         1
*DATABASE_ELOUT
$#      dt    binary      lcur     ioopt
      0.05         2         0         1
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid
       0.5         0         0         0         0
$#   ioopt
         0
*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         0         0         3         0         1         1         1         1
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         0         0         1         1         1         2         1
$# nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout
         0         0       1.0         0         0         0    STRESS    STRESS
$#    dtdt    resplt     neipb
         0         0         0
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$                                   Contact                                   $$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
*CONTACT_AUTOMATIC_SINGLE_SURFACE
$#    ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr
         0         0         0         0         0         0         0         0
$#      fs        fd        dc        vc       vdc    penchk        bt        dt
      0.08       0.8       0.0       0.0       0.0         0       0.0    1.0E20
$#     sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf
       0.0       1.0       0.0       0.0       1.0       1.0       1.0       1.0
$
*END