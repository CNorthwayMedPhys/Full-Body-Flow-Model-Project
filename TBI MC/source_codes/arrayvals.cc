#include <iostream>
#include <fstream>

using namespace std;

int main()
{

  float start;
  float end;
  float dy;
  char filename[12];

  int numvals;

  cout << "Start value: " << endl;
  cin >> start;

  cout << "End value: " << endl;
  cin >> end;

  cout << "Spacing (probably 0.25): " << endl;
  cin >> dy;

  cout << "Filename (format yvals01P.txt)" << endl;
  cin >> filename;

  numvals = ((end-start)/dy)+1;

  ofstream myfile (filename);
  if (myfile.is_open())
    {
    
    for (int i=0; i < numvals; i++)
      {
        myfile << start+dy*i;
        myfile << " ";
      }
        
    myfile.close();
    }
  cout << "Done" << endl;

  return 0;

}
