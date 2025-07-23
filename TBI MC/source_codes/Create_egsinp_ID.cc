/**********************************************************
*		Author and copyright: Tony Teke		  *
***********************************************************/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define OK 0
#define FAIL 1
using namespace std;


int main()
{
int aux=0;
// TO CHANGE THIS IS NOT WORKING
char *user="tbicobalt";
float ecut=0.01; 
float pcut=0.7;

int zeroairdose=0;
int doseprint=0;
int MAX20=0;
int num_indexes=21;

// Updated the coordinates of Cx,Cy,Cz with Conrad's updated data for Rando TBI
float SSD=153.;
float Cx=-0.14;
float Cy=-12.89-SSD;
float Cz=-18.;

float xiso=Cx;
float yiso=0.;
float ziso=0.;

float theta=50.;
float phi=270.;
float phicol=0.;
float index=0.;

float Pi=3.14159265358;
float Radius=SSD-45.; // Or distance from phase space scoring plane to isocenter

char *outputFileName="12345678_AP_complete.egsinp";
// open output egsinp file
FILE *oStream;
if( (oStream = fopen(outputFileName,"w") ) == NULL )      
{
  printf("\n ERROR: can't open file %s! \n", outputFileName);
  return(FAIL);
}

fprintf(oStream,"Cobalt Sweeping Beam for TBI \t \t \t #!GUI1.0\n");
fprintf(oStream,"%d\n",aux);
// have updated here the path to the phantom file to match what it is on the gundam cluster
fprintf(oStream,"/home/%s/MC/0620412/12345678_AP_complete.egsphant\n",user);
fprintf(oStream,"%1.1f, %1.2f, %d\n",pcut,ecut,aux);
fprintf(oStream,"%d, %d, %d\n",zeroairdose,doseprint,MAX20);
// TO CHANGE THE NEXT LINE AT SOME POINT
fprintf(oStream,"2, 21, %d, 1, 0\n", num_indexes);
	//Changed to debug source12
	yiso=Cy+SSD*cos( ((90.-theta)*Pi/180.) );
	ziso=Cz-SSD*sin( ((90.-theta)*Pi/180.) );
	//yiso=Cy+Radius*cos( ((90.-theta)*Pi/180.) );
	//ziso=Cz-Radius*sin( ((90.-theta)*Pi/180.) );

for(int i=1;i<=num_indexes;i++)
{

	fprintf(oStream,"%f, %f, %f, %.1f, %.1f, %.1f, %.1f, %f\n",xiso,yiso,ziso,theta,phi,phicol,Radius,index);
	//section changing coord and values
	theta=50.+4.5*i;
/*	if(theta>360)
	{
		theta-=360;
	}*/
	yiso=Cy+SSD*cos( ((90.-theta)*Pi/180.) );
	ziso=Cz-SSD*sin( ((90.-theta)*Pi/180.) );
	index+=1/(static_cast<float>(num_indexes-1));
}
// TO ADD THE REMAINING OF THE INPUT FILE
srand(time(0));
//(rand()%1000)
fprintf(oStream,"2, 2, 0, 150, 0, 0, 0, 0\n");
fprintf(oStream,"BEAM_Cobalt,cobalt10,700icru_TT,0,0\n");
//fprintf(oStream,"1000000000, 0, 999, 33, 97, 100.0, 0, 0, 0, 0, , 0, 0, 0, 1, 0\n");
 fprintf(oStream,"2000000000, 0, 999, %d, %d, 100.0, 0, 0, 1, 0, , 0, 0, 0, 1, 0\n",rand()%1000,rand()%1000 );
 
 fprintf(oStream,"#########################\n");
 fprintf(oStream,":Start MC Transport Parameter:\n\n");
 fprintf(oStream,"Global ECUT= 0.7\n");
 fprintf(oStream,"Global PCUT= 0.01\n");
 fprintf(oStream,"Global SMAX= 5\n");
 fprintf(oStream,"ESTEPE= 0.25\n");
 fprintf(oStream,"XIMAX= 0.5\n");
 fprintf(oStream,"Boundary crossing algorithm= PRESTA-I\n");
 fprintf(oStream,"Skin depth for BCA= 0\n");
 fprintf(oStream,"Electron-step algorithm= PRESTA-II\n");
 fprintf(oStream,"Spin effects= On\n");
 fprintf(oStream,"Brems angular sampling= Simple\n");
 fprintf(oStream,"Brems cross sections= BH\n");
 fprintf(oStream,"Bound Compton scattering= Off\n");
 fprintf(oStream,"Pair angular sampling= Simple\n");
 fprintf(oStream,"Photoelectron angular sampling= Off\n");
 fprintf(oStream,"Rayleigh scattering= Off\n");
 fprintf(oStream,"Atomic relaxations= Off\n");
 fprintf(oStream,"Electron impact ionization= Off\n\n");
 fprintf(oStream,":Stop MC Transport Parameter:\n");
 fprintf(oStream,"#########################\n");
return(OK);

}
