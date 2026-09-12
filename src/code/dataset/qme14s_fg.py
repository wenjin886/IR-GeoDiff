# Define SMARTS
alkane               = "[CX4;H3,H2,H1]"
alkene               = "[CX3]=[CX3]"
enol                 = "[CX3]=[CX3]([OH])"
enamine              = "[CX3]=[CX3]([NX3])"
alkyne               = "[CX2]#[CX2]"
amine                = "[NX3;!$(NC=O)]"
hydrazine            = "[NX3]-[NX3]"
imine                = "[NX2]=[CX3]"
azo_compound         = "[NX2]=[NX2]"
nitrile              = "[NX1]#[CX2]"
alcohol              = "[OX2H;!$(OC=O)]"
ether                = "[OX2H0;!$(OC=O);!$([O]-[O])]"
aldehyde             = "[#6,H][CX3H1](=O)"
ketone               = "[#6][CX3](=O)[#6]"
carboxylic_acid      = "[CX3](=O)[OX2H1]"
ester                = "[CX3](=O)[OX2H0]"
amide                = "[CX3](=O)[NX3]"
acid_anhydride       = "[CX3](=O)O[CX3](=O)"
imide                = "[CX3](=O)[NX3][CX3](=O)"
carbamate            = "[OX2][CX3](=O)[NX3]"
isocyanate           = "[NX2]=[CX2](=O)"
arene                = "[$([cX2](:*):*),$([cX3](:*):*)]"     
imidazole            = "[#7]:[#6]:[#7]"
pyrazole             = "[#7]:[#7]"
oxazole              = "[#7]:[#6]:[#8]"
isoxazole            = "[#7]:[#8]"
cyclopropane         = "C1CC1"
epoxide              = "C1OC1"
acyl_halide          = "[!#1](=[O,S,Se])[#9,#17,#35]"
haloalkane           = "[!#1;!$(C(=[O,S,Se])[#9,#17,#35])][#9,#17,#35]"
thiol                = "[SX2H]"
sulfide              = "[SX2H0]"
thiol_sulfide        = "[SX2]"
thial                = "[CX3H1](=[S,Se])"
thioketone           = "[CX3H0](=[S,Se])"
thial_thioketone     = "[CX3](=[S,Se])"
thioamide            = "[NX3][CX3](=[S,Se])"
isothiocyanate       = "[NX2]=[CX2](=[S,Se])"
sulfoxide            = "[SX3]=[O,Se]"
sulfone              = "[SX4](=[O,Se])(=[O,Se])"
sulfonate            = "[OX2][SX4](=[O,Se])(=[O,Se])"
sulfonic_compound    = "[OX2][SX4](=[O,Se])(=[O,Se])[OX2]"
sulfonamide          = "[NX3][SX4](=[O,Se])(=[O,Se])"




fg_smart_list = [
    alkane           , 
    alkene           , 
    enamine          , 
    alkyne           , 
    amine            , 
    imine            , 
    nitrile          , 
    alcohol          , 
    ether            , 
    aldehyde         , 
    ketone           , 
    carboxylic_acid  , 
    ester            , 
    amide            , 
    carbamate        , 
    arene            , 
    imidazole        , 
    pyrazole         , 
    oxazole          , 
    isoxazole        , 
    cyclopropane     , 
    epoxide          , 
    haloalkane       , 
    thiol_sulfide ,
    sulfone          , 
    sulfonate        , 
]

fg_name_list = [
    "alkane"           , 
    "alkene"           , 
    "enamine"          , 
    "alkyne"           , 
    "amine"            , 
    "imine"            , 
    "nitrile"          , 
    "alcohol"          , 
    "ether"            , 
    "aldehyde"         , 
    "ketone"           , 
    "carboxylic_acid"  , 
    "ester"            , 
    "amide"            , 
    "carbamate"        , 
    "arene"            , 
    "imidazole"        , 
    "pyrazole"         , 
    "oxazole"          , 
    "isoxazole"        , 
    "cyclopropane"     , 
    "epoxide"          , 
    "haloalkane"       , 
    "thiol_sulfide"    ,
    "sulfone"          , 
    "sulfonate"        , 
]

