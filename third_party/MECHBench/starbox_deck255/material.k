$#  units:  kg  mm  ms  kN  GPa  kN-mm
*KEYWORD
*MAT_PIECEWISE_LINEAR_PLASTICITY
$#     mid        ro         e        pr      sigy      etan      fail      tdel
       999  7.83e-06     200.0       0.3     0.366       0.0     1e+21       0.0
$#       c         p      lcss      lcsr        vp
        40       5.0         1         0         1
$#    eps1      eps2      eps3      eps4      eps5      eps6      eps7      eps8
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$#     es1       es2       es3       es4       es5       es6       es7       es8
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
*DEFINE_CURVE
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp
         1         0       1.0       1.0       0.0       0.0         0
$#                a1                  o1
                 0.0               0.366
               0.025               0.424
               0.049               0.476
               0.072               0.507
               0.095               0.529
               0.118               0.546
                0.14               0.559
               0.182               0.584
*END