# Define SMARTS
alkane          = "[CX4;H3,H2,H1]"      
alkene          = "[CX3]=[CX3]"      
alkyne          = "[CX2]#[CX2]"   

amine           = "[NX3;!$(NC=O)]"
imine           = "[NX2]=[CX3]"  
nitrile         = "[NX1]#[CX2]"      

alcohol         = "[OX2H;!$(OC=O)]"     
ether           = "[OX2H0;!$(OC=O);!$([O]-[O])]"  

haloalkane      = "[#6;!$(C(=O)[F])][F]"    

aldehyde        = "[#6,H][CX3H1](=O)"
ketone          = "[#6][CX3](=O)[#6]"
ester           = "[CX3](=O)[OX2H0]"
amide           = "[CX3](=O)[NX3]"

arene           = "[$([cX2](:*):*),$([cX3](:*):*)]"
imidazole       = "[#7]:[#6]:[#7]" 
pyrazole        = "[#7]:[#7]"
oxazole         = "[#7]:[#6]:[#8]"
isoxazole       = "[#7]:[#8]"

cyclopropane    = "C1CC1"
epoxide         = "C1OC1"




fg_smart_list = [
        alkane,
        alkene,
        alkyne,
        
        amine, 
        imine,
        nitrile,

        alcohol,
        ether,
        
        haloalkane, 

        aldehyde,
        ketone, 
        ester, 
        amide, 

        arene, 
        imidazole,
        pyrazole,
        oxazole,
        isoxazole,

        cyclopropane,    
        epoxide, 


        ]

fg_name_list = [
        "alkane",
        "alkene",
        "alkyne",

        "amine", 
        "imine",
        "nitrile",

        "alcohol",
        "ether", 

        "haloalkane", 

        "aldehyde",
        "ketone", 
        "ester", 
        "amide", 

        "arene", 
        "imidazole",
        "pyrazole",
        "oxazole",
        "isoxazole",
        
        "cyclopropane",    
        "epoxide",    
 
        
                  
    
        ]

# if __name__ == "__main__":
#     fg = 

