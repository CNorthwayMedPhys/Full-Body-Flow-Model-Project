/*      Savitzky-Golay Adaptive Filter for MC .3ddose Files
	I.Thomson/A.Bergman   June 2005
*/

// usage : SGolay3d some.3ddose max_window_size
// Note: Kawrakow used a 7x7x7 in his paper  Med.Phys.Biol.47(2002)3087-3103

		
#include <stdio.h>	//	for printf,fopen,fclose,fscanf,fprintf
#include <stdlib.h>	//	for malloc,free
#include <math.h>	//	for sqrt
#include <assert.h>
#include <time.h>
#include <algorithm>
#include <string.h> 

struct _3ddose_data_t
{
	int x_size, y_size, z_size;
	double* x_edges;
	double* y_edges;
	double* z_edges;
	double* dosedata;
	double* errordata;
};

double* read_doublearray( FILE* filehandle, int count )
{
	int i;
	double* data;

	//	allocate memory for read
	data = (double*)malloc( sizeof(double)*count );
	
	if(data==NULL)
		printf("Could not allocate memory for %d doubles\n", count);
	//	read in the doubles
	for ( i=0; i<count; ++i )
	{
		if ( fscanf( filehandle, "%lf", &data[i] ) != 1 )
		{
			printf("Could not read values from file. %d of %d doubles read\n", i, count);
			free( data );
			return NULL;
		}
	}

	return data;
}

void write_doublearray( FILE* filehandle, double* data, int count )
{
	int i;
	int wrap = 0;
	for ( i=0; i<count; ++i )
	{
		if ( (data[i] != 0.0) && ((fabs(data[i]) < 1e-6) || (fabs(data[i]) > 1e6)) )
		{
			fprintf( filehandle, "%.9lE ", data[i] );
		}
		else
		{
			fprintf( filehandle, "%.7lf ", data[i] );
		}

		if ( ++wrap == 5 )
		{
			fprintf( filehandle, "\n" );
			wrap = 0;
		}
	}
	if ( wrap != 0 )
		fprintf( filehandle, "\n" );
}


void free_3ddosedata( struct _3ddose_data_t* dosedata )
{
	if ( dosedata->x_edges != NULL )
	{
		free( dosedata->x_edges );
		dosedata->x_edges = NULL;
	}
	if ( dosedata->y_edges != NULL )
	{
		free( dosedata->y_edges );
		dosedata->y_edges = NULL;
	}
	if ( dosedata->z_edges != NULL )
	{
		free( dosedata->z_edges );
		dosedata->z_edges = NULL;
	}
	if ( dosedata->dosedata != NULL )
	{
		free( dosedata->dosedata );
		dosedata->dosedata = NULL;
	}
	if ( dosedata->errordata != NULL )
	{
		free( dosedata->errordata );
		dosedata->errordata = NULL;
	}
}

/***
read_3ddosefile : reads in a 3ddose file from 'filename' into 'dosedata'.
return value: 0 on success, -1 for failure
***/
int read_3ddosefile( char* filename, struct _3ddose_data_t* dosedata )
{
	FILE* filehandle;
	
	//	initialize dosedata
	dosedata->x_size = 0;
	dosedata->y_size = 0;
	dosedata->z_size = 0;
	dosedata->x_edges = NULL;
	dosedata->y_edges = NULL;
	dosedata->z_edges = NULL;
	dosedata->dosedata = NULL;
	dosedata->errordata = NULL;

	//	open the dose file
	filehandle = fopen( filename, "rt" );
	if ( filehandle == NULL )
	{	
		printf("Filehandle is NULL\n");
		return -1;
	}

	//	read in the size of the dose array
	if ( fscanf( filehandle, "%d %d %d", &dosedata->x_size, &dosedata->y_size, &dosedata->z_size ) != 3 )
	{
		printf("Something is wrong with the sixe of the dose array\n");
		fclose( filehandle );
		return -1;
	}

	//	read in the dose boundaries
	dosedata->x_edges = read_doublearray( filehandle, dosedata->x_size+1 );
	dosedata->y_edges = read_doublearray( filehandle, dosedata->y_size+1 );
	dosedata->z_edges = read_doublearray( filehandle, dosedata->z_size+1 );
	if (( dosedata->x_edges == NULL ) || ( dosedata->y_edges == NULL ) || ( dosedata->z_edges == NULL ))
	{
		free_3ddosedata( dosedata );
		fclose( filehandle );
		printf("Something is wrong with the dose boundaries\n");
		return -1;
	}

	//	read in the dose values
	dosedata->dosedata = read_doublearray( filehandle, dosedata->x_size * dosedata->y_size * dosedata->z_size );
	if ( dosedata->dosedata == NULL )
	{
		free_3ddosedata( dosedata );
		fclose( filehandle );
		printf("Something is wrong with the dose values\n");
		return -1;
	}

	//	read in the error values
	dosedata->errordata = read_doublearray( filehandle, dosedata->x_size * dosedata->y_size * dosedata->z_size );
	if ( dosedata->errordata  == NULL )
	{
		free_3ddosedata( dosedata );
		fclose( filehandle );
		printf("Something is wrong with the error values\n");
		return -1;
	}

	fclose( filehandle );
	return 0;
}

int write_3ddosefile( char* filename, struct _3ddose_data_t* dosedata )
{
	int count;
	FILE* filehandle;

	filehandle = fopen( filename, "wt" );
	if ( filehandle == NULL )
		return -1;

	fprintf( filehandle, "%d %d %d\n", dosedata->x_size, dosedata->y_size, dosedata->z_size );

	write_doublearray( filehandle, dosedata->x_edges, dosedata->x_size+1 );
	write_doublearray( filehandle, dosedata->y_edges, dosedata->y_size+1 );
	write_doublearray( filehandle, dosedata->z_edges, dosedata->z_size+1 );

	count = dosedata->x_size * dosedata->y_size * dosedata->z_size;
	write_doublearray( filehandle, dosedata->dosedata, count );
	write_doublearray( filehandle, dosedata->errordata, count );

	fclose( filehandle );

	return 0;
}

double getdose( struct _3ddose_data_t* dosedata, int x, int y, int z )
{
	int idx = (x-1) + (y-1)*dosedata->x_size + (z-1)*(dosedata->x_size*dosedata->y_size);
	return dosedata->dosedata[idx];
}

void setdose( struct _3ddose_data_t* dosedata, int x, int y, int z, double dose )
{
	int idx = (x-1) + (y-1)*dosedata->x_size + (z-1)*(dosedata->x_size*dosedata->y_size);
	dosedata->dosedata[idx] = dose;
}

void SGolay3D( struct _3ddose_data_t* dosedata, int max_window_size )
{
	time_t starttime = time(NULL);
	
	int length_x = dosedata->x_size;
	int length_y = dosedata->y_size;
	int length_z = dosedata->z_size;


	double* tdd = dosedata->dosedata;
	double* tddsmooth = new double[ length_x*length_y*length_z ];
	double* unc = new double[ length_x*length_y*length_z ];
	double* uncsmooth = new double [ length_x*length_y*length_z ];
	for ( int i=0; i<length_x*length_y*length_z; ++i )
	{
		tddsmooth[i] = tdd[i];
		unc[i] = tdd[i] * dosedata->errordata[i]; 
		uncsmooth[i] = tdd[i] * dosedata->errordata[i]; //LB added Jan 11 2018, or else they're zero at the end if voxel not smoothed
		/*if (tdd[i] == 0.000)
		{
			printf("tdd = %e\n",tdd[i]);
		}
		if ( unc[i] == 0.0000 )
		{
			printf("location i: %d  tdd= %e  dosedata= %f  unc = %f\n",i, tdd[i], dosedata->errordata[i],unc[i]);
			//unc[i] =  2.22e-16;	//	MATLAB's 'eps' (epsilon)
		}
		*/
	}

	int nmax_i = max_window_size;
	printf("length_x = %d\t",length_x);
	printf("nmax_i = %d\n",nmax_i);

	int nmax_j = max_window_size;
	printf("length_y = %d\t",length_y);
	printf("nmax_j = %d\n",nmax_j);

	int nmax_k = max_window_size;
	printf("length_z = %d\t",length_z);
	printf("nmax_k = %d\n",nmax_k);
	

	for ( int  i=1; i<=length_x; ++i )  
	{
		for ( int j=1; j<=length_y; ++j )  
		{
			for (int k=1; k<=length_z; ++k )
			{
				int n_i, n_j, n_k;
				double g_i, g_j, g_k;
				g_i = 0;
				g_j = 0;
				g_k = 0;

				if ( i < 3 )
				{
					n_i = 0;
				}
				else if ( i-1 < nmax_i )
				{
					n_i = i-1;
				}
				else if ( length_x-i < nmax_i )
				{
					n_i = length_x-i;
				}
				else if ( length_x-i < 3 )
				{
					n_i = 0;
				}
				else
				{
					n_i = nmax_i;
				}

				if ( j < 3 )
				{
					n_j = 0;
				}
				else if ( j-1 < nmax_j )
				{
					n_j = j-1;
				}
				else if ( length_y-j < nmax_j )
				{
					n_j = length_y-j;
				}
				else if ( length_y-j < 3 )
				{
					n_j = 0;
				}
				else
				{
					n_j = nmax_j;
				}

				if ( k < 3 )
				{
					n_k = 0;
				}
				else if ( k-1 < nmax_k )
				{
					n_k = k-1;
				}
				else if ( length_z-k < nmax_k )
				{
					n_k = length_z-k;
				}
				else if ( length_z-k < 3 )
				{
					n_k = 0;
				}
				else
				{
					n_k = nmax_k;
				}


				while ( n_i >= 2 )
				{
					g_i=n_i*(n_i+1)/3.0;

					double b0, b1, b2;
					b0=0; b1=0; b2=0;

					for ( int ip=-n_i; ip<=n_i; ++ip )
					{
						double temp = (1/(double)(2*n_i+1)) * getdose( dosedata, i+ip, j, k );

						b0 += temp * (1-(5*((ip*ip)-g_i)/(4*g_i-1)));
						b1 += temp * (ip/g_i);
						b2 += temp * (5*((ip*ip)-g_i)/(g_i*(4*g_i-1)));
					}
				
					double chi_square = 0;
					for ( int ip=-n_i; ip<=n_i; ++ip )
					{
						int idx = (i+ip-1) + (j-1)*dosedata->x_size + (k-1)*(dosedata->x_size*dosedata->y_size);

						double chi = (b0+b1*ip+b2*ip*ip-tdd[idx])/unc[idx];
						chi_square += chi*chi;
					}
					double chi_test = chi_square/(2*n_i-2);  // np=3 1D -> (2*n_i+1  - np) -> (2*n_i-2) 
					if ( chi_test<1 )
						break;	//	break out of while loop
					
					n_i=n_i-1;
					g_i=n_i*(n_i+1)/3.0;
				}

				while ( n_j >= 2 )
				{


					g_j=n_j*(n_j+1)/3.0;
					double b0, b1, b2;
					b0=0; b1=0; b2=0;



					for ( int jp=-n_j; jp<=n_j; ++jp )
					{
						double temp = (1/(double)(2*n_j+1)) * getdose( dosedata, i, j+jp, k );

						b0 += temp * (1-(5*((jp*jp)-g_j)/(4*g_j-1)));
						b1 += temp * (jp/g_j);
						b2 += temp * (5*((jp*jp)-g_j)/(g_j*(4*g_j-1)));
					}


					double chi_square = 0;
					//for ( int jp=-n_j; jp<=n_j; ++jp )  
					for ( int jp=-n_j; jp<=n_j; ++jp )
					{
						int idx = (i-1) + (j+jp-1)*dosedata->x_size + (k-1)*(dosedata->x_size*dosedata->y_size);

						double chi = (b0+b1*jp+b2*jp*jp-tdd[idx])/unc[idx];
						chi_square += chi*chi;
					}
					double chi_test=chi_square/(2*n_j-2);  //should be (chi_square/(2*n_j+1-np))==(chi_square/(2*n_j+1-2<--np??4 or 2?))??
					if ( chi_test<1 )
						break;	//	break out of while loop
					

				    
					n_j=n_j-1;
					g_j=n_j*(n_j+1)/3.0;


				}


				while ( n_k >= 2 )
				{


					g_k=n_k*(n_k+1)/3.0;
					double b0, b1, b2;
					b0=0; b1=0; b2=0;



					for ( int kp=-n_k; kp<=n_k; ++kp )
					{
						double temp = (1/(double)(2*n_k+1)) * getdose( dosedata, i, j, k+kp );

						b0 += temp * (1-(5*((kp*kp)-g_k)/(4*g_k-1)));
						b1 += temp * (kp/g_k);
						b2 += temp * (5*((kp*kp)-g_k)/(g_k*(4*g_k-1)));
					}


					double chi_square = 0;
					//for ( int kp=-n_k; kp<=n_k; ++kp )  //AB removed "int" redefinition
					for ( int  kp=-n_k; kp<=n_k; ++kp )
					{
						int idx = (i-1) + (j-1)*dosedata->x_size + (k+kp-1)*(dosedata->x_size*dosedata->y_size);

						double chi = (b0+b1*kp+b2*kp*kp-tdd[idx])/unc[idx];
						chi_square += chi*chi;
					}
					double chi_test=chi_square/(2*n_k-2);
					if ( chi_test<1 )
						break;	//	break out of while loop
	
					//printf("n_k=%d  chi_test=%f k=%d - making window smaller\n",n_k,chi_test,k);
					n_k=n_k-1;
					g_k=n_k*(n_k+1)/3.0;


				}
			



				
				if ( n_i < 2 )
				{
					n_i=0;
					g_i=0;
				}
				if ( n_j < 2 )
				{
					n_j=0;
					g_j=0;
				}
				if ( n_k < 2 )
				{
					n_k=0;
					g_k=0;
				}


				
				int Ni=(2*n_i+1);  
				int Nj=(2*n_j+1);
				int Nk=(2*n_k+1);
				int Nijk=Ni*Nj*Nk;


/*
				%If Ni, Nj, or Nk are equal to one, the corresponding matrix of window lengths (Nis,Njs,Nks) will be empty.

				%If any of Nis,Njs,or Nks are empty, assign them the value 1, otherwise, the length of the empty matrices will be zero,
				%and the for loop starting on line 129 will not initialize the 3d matrix of possible window sizes (Nijks), even if
				%there are possible window sizes other than the one calculated one dimension at a time.
				
*/
				int Nis[125], Njs[125], Nks[125];
				int i_size = ((Ni-2)+1)/2;
				if ( i_size == 0 )
				{
					Nis[0] = 1;
					i_size = 1;
				}
				else
				{
					for ( int ii=0; ii<i_size; ++ii )
					{
						Nis[ii] = Ni-2 + (-2*ii);
					}
				}

				int j_size = ((Nj-2)+1)/2;
				if ( j_size == 0 )
				{
					Njs[0] = 1;
					j_size = 1;
				}
				else
				{
					for ( int jj=0; jj<j_size; ++jj )
					{
						Njs[jj] = Nj-2 + (-2*jj);
					}
				}

				int k_size = ((Nk-2)+1)/2;
				if ( k_size == 0 )
				{
					Nks[0] = 1;
					k_size = 1;
				}
				else
				{
					for ( int kk=0; kk<k_size; ++kk )
					{
						Nks[kk] = Nk-2 + (-2*kk);
					}
				}

/*
				Nijks=[]; %Ensure that Nijks is empty before it is initialized for each voxel.
				if Nijk>=5 %Only check 3-dimensional chi-square if window is large enough for smoothing.
*/

				int pass=0;
				double b000=0; 
				double b000_good = 0;
				int b000_onecount = 4;
				
 				if ( Nijk >= 5 )
				{
					assert( i_size <= 10 );
					assert( j_size <= 10 );
					assert( k_size <= 10 );

					double Nijks[10*10*10];
					for ( int r=0; r<i_size; ++r )
					{
						for ( int s=0; s<j_size; ++s )
						{
							for ( int t=0; t<k_size; ++t )
							{
								int idx = r + s*i_size + t*(i_size*j_size);
								Nijks[idx] = Nis[r]*Njs[s]*Nks[t];
							}
						}
					}
					double Nijksrow[10*10*10];
					std::copy( Nijks, Nijks+(i_size*j_size*k_size), Nijksrow );
					std::sort( Nijksrow, Nijksrow+(i_size*j_size*k_size) );
					std::reverse( Nijksrow, Nijksrow+(i_size*j_size*k_size) );


					int w=0;
					double* Nijks_search = Nijks;
					for ( w=0; true; ++w )
					{
						b000=0;
						double b001=0; double b010=0; double b100=0; double b011=0; double b110=0; double b101=0; 
						double b002=0; double b020=0; double b200=0;

						for ( int ip=-n_i; ip<=n_i; ++ip )
						{
							for ( int jp=-n_j; jp<=n_j; ++jp )
							{
 								for ( int kp=-n_k; kp<=n_k; ++kp )
								{
									double tdd = getdose( dosedata, i+ip,j+jp,k+kp );
									//printf("tdd = %.8E\t",tdd);
									b000=b000+tdd*(1-5*(ip*ip-g_i)/(4*g_i-1)-5*(jp*jp-g_j)/(4*g_j-1)-5*(kp*kp-g_k)/(4*g_k-1));
									
									if ( g_k != 0.0 ) //AB 13July05 - changed "0" to "0.0"
									{
										b001 += tdd*kp/g_k;
										b002 += tdd*5*(kp*kp-g_k)/(g_k*(4*g_k-1));
										if (n_k == 0)
										{
											printf("g_k not = 0 but n_k is! n_k=%d g_k=%f, ijk=%d,%d,%d\n", n_k,g_k,i,j,k);
										}
									}
									if ( g_j != 0.0 )
									{
										b010 += tdd*jp/g_j;
										b020 += tdd*5*(jp*jp-g_j)/(g_j*(4*g_j-1));
										if (n_j == 0)
										{
											printf("g_j not = 0 but n_j is! n_j=%d  g_j=%f ijk=%d,%d,%d\n", n_j,g_j,i,j,k);
										}
									}
									if ( g_i != 0.0 )
									{
										b100 += tdd*ip/g_i;
										b200 += tdd*5*(ip*ip-g_i)/(g_i*(4*g_i-1));
										if (n_i == 0)
										{
											printf("g_i not = 0 but n_i is! n_i=%d  g_i=%fijk=%d,%d,%d\n", n_i,g_i,i,j,k);
										}
									}

									if ( (g_j!=0.0) && (g_k!=0.0) )
										b011 += tdd*jp*kp/(g_j*g_k);
									if ( (g_i!=0.0) && (g_j!=0.0) )
										b110 += tdd*ip*jp/(g_i*g_j);
									if ( (g_i!=0.0) && (g_k!=0.0) )
										b101 += tdd*ip*kp/(g_i*g_k);
								}
							}
						}

						Ni=(2*n_i+1);
						Nj=(2*n_j+1);
						Nk=(2*n_k+1);
						Nijk=Ni*Nj*Nk;  //** this is the last instance of Nijk... use to calc uncertainty?
						b000=b000/Nijk;
						b001=b001/Nijk;
						b010=b010/Nijk;
						b100=b100/Nijk;
						b011=b011/Nijk;
						b110=b110/Nijk;
						b101=b101/Nijk;
						b002=b002/Nijk;
						b020=b020/Nijk;
						b200=b200/Nijk;

						//if (b000 > 3.0)
						//{
						//	printf("b000 biiiig! WHY?  b000= %f  Nijk=%d  i,j,k=%d,%d,%d   n_i,n_j,n_k=%d,%d,%d\n",b000, Nijk, i,j,k, n_i,n_j,n_k);
						//	i=i;
						//}
	
						double chi_square=0;
						for (int  ip=-n_i; ip<=n_i; ++ip ) 
						{
							for ( int jp=-n_j; jp<=n_j; ++jp )
							{
								for ( int kp=-n_k; kp<=n_k; ++kp )
								{
									int idx = (i+ip-1) + (j+jp-1)*dosedata->x_size + (k+kp-1)*(dosedata->x_size*dosedata->y_size);

									double chi = (b000 + 
												b001*kp + b010*jp + b100*ip + 
												b011*jp*kp + b110*ip*jp + b101*ip*kp + 
												b002*kp*kp + b020*jp*jp + b200*ip*ip - tdd[idx]) / unc[idx];
									chi_square += chi*chi;
								}
							}
						}


						int zeros = 0;
						if ( n_i == 0 )
							++zeros;
						if ( n_j == 0 )
							++zeros;
						if ( n_k == 0 )
							++zeros;

						int n_p = 0;
						if ( zeros == 0 )
							n_p = 10;  
						else if ( zeros == 1 )
							n_p = 6;
						else if ( zeros == 2 )
							n_p = 3;
						else
							n_p = 0;
						if (n_p == 0)
						{
							printf("n_p = %d !!\n\n",n_p);
						}


						double chi_test=chi_square/(Nijk-n_p);
						if ( chi_test >= 1 )
						{
							//printf("Warning: Chi test failed in b001,b010 zone: i,j,k=%d,%d,%d  w=%d Nijksrow[w]=%f\n",i,j,k,w,Nijksrow[w]);

							//	if window is less than 5, break from the big loop
							if ( Nijksrow[w]<5 )
								break;

							//	find the 'w'th element's location in the 'Nijks' matrix
							int row, col;

							//	If the last one doesn't match, restart the search
							if ( Nijksrow[w] != Nijksrow[w-1] )
								Nijks_search = Nijks;

							double* found_offset = std::find( Nijks_search, Nijks+(i_size*j_size*k_size), Nijksrow[w] ); 
							// AB ?? found offset looks the same as Nijks
							
							//	Start search at entry after the one found
							Nijks_search = found_offset+1;

							//	Calculate the offset in 'Nijks'
							int idx = (int)std::distance(Nijks,found_offset);

							//	get the row and column
							row = idx%i_size;
							col = idx/i_size;

							int r, t, s;
							r=row;
							t=(int)(floor(col/(float)j_size));
							s=col-j_size*t;
							assert( r < i_size );
							assert( s < j_size );
							assert( t < k_size );
							Ni=Nis[r];
							Nj=Njs[s];
							Nk=Nks[t];
							assert( Ni >= 0 );
							assert( Nj >= 0 );
							assert( Nk >= 0 );
							n_i=(Ni-1)/2;
							n_j=(Nj-1)/2;
							n_k=(Nk-1)/2;
							g_i=n_i*(n_i+1)/3.0;  //AB Jul05 - added because otherwise n_i decrements but g_i doesn't change
							g_j=n_j*(n_j+1)/3.0;
							g_k=n_k*(n_k+1)/3.0;
						}	
						else
						{
							//	Check if we have a matching value above or below
							if ( ((w>0) && (Nijksrow[w] == Nijksrow[w-1])) ||
								((w<(i_size*j_size*k_size)-1) && (Nijksrow[w] == Nijksrow[w+1])) )
							{
								pass=pass+1;

								int onecount = 0;
								if ( Ni == 1 )
									++onecount;
								if ( Nj == 1 )
									++onecount;
								if ( Nk == 1 )
									++onecount;
								if ( onecount < b000_onecount )
								{
									b000_onecount = onecount;
									b000_good = b000;
								}

								if ( Nijksrow[w] != Nijksrow[w+1] )
									break;
							}
							else
							{
//								printf( "n_i=%d n_j=%d n_k=%d\n", n_i, n_j, n_k );
								break;
							}
						}
					}
				}



				if ( Nijk >= 5 ) // 1x1x5 and 3x3x3 are both okay
				{
					int idx = (i-1) + (j-1)*dosedata->x_size + (k-1)*(dosedata->x_size*dosedata->y_size);
					uncsmooth[idx]=sqrt((unc[idx]*unc[idx])/Nijk);  //pg 3089 Kawrakow paper.  May need to be checked.

					if ( pass > 0 )
					{
						tddsmooth[idx] = b000_good;
						
					}
					else
					{
						tddsmooth[idx] = b000;
						
					}

					
				}


			
			}
			
		}
		
       
	printf( " %d of %d \n", i,length_x);
	time_t endtime = time(NULL);
	int timediff = endtime - starttime; 
	//printf(" Time to run: %d: %02d\n",timediff/60,timediff%60);

	float avg_time_pervoxel = timediff / (float)i; 
	int remaining_time = avg_time_pervoxel * (length_x - i);
	
	printf("Estimated time remaining = %d:%02d\n",remaining_time/60, remaining_time%60);
	}
	time_t endtime = time(NULL);
	int timediff = endtime - starttime;
	printf(" time to run: %d:%02d\n", timediff/60, timediff%60);

	for (int i=0; i<length_x*length_y*length_z; ++i )   //AB removed "int" redefinition
	{
		dosedata->dosedata[i] = tddsmooth[i];
		if (tddsmooth[i] > 0)
		{ 
		    dosedata->errordata[i] = uncsmooth[i]/tddsmooth[i];
		}
		
	}

	delete [] tddsmooth;
	delete [] unc;

}

int main( int argc, char* argv[] )
{
	
	char* filename;
	char newfilename[255];
	struct _3ddose_data_t dosedata;
	int max_window_size;

	if ( argc < 3 )
	{
		printf( "usage : SGolay3D <Please enter .3ddose file to filter as 1st argument> <Max window size - usually 7>\n" );
		return -1;
	}

	//	get the wildcard name to load from
	filename = argv[1];

	max_window_size = atoi(argv[2]);

	
	//	load the first 3ddose file
	printf("Loading in .3ddose file ....\n");
	if ( read_3ddosefile( filename, &dosedata ) == -1 )
	{
		printf("Can't find filename - Please try again.");
		return -1;
	}

	printf("Starting Savitzky-Golay filter...\n");
	SGolay3D(&dosedata,max_window_size);
	
	//	write our resulting dose file
	strcpy( newfilename, "sgolay_" );
	strcat( newfilename, filename );
	write_3ddosefile( newfilename, &dosedata );

	//	free the dose memory
	free_3ddosedata( &dosedata );
	printf("\n");
	return 0;
}
