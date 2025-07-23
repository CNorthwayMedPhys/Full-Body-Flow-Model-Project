#include <iostream>
#include <fstream>

using namespace std;

//This assumes the plastic tray is 0.6cm. 

int main()
{

  float bed_y, t0_y;
  int numstys;
  float tray_thick;
  float sty_thick = 3.7;
  float shift_y_plastic, shift_y_lead, SSD;
  float sep = 44.8;
  
  cout << "bed_y value from egsphant: " << endl;
  cin >> bed_y;

  cout << "t0_y from egsphant: " << endl;
  cin >> t0_y;

  cout << "Number of styros: " << endl;
  cin >> numstys;

  cout << "Tray thickness (normally 0.6): " << endl;
  cin >> tray_thick;

  shift_y_plastic = bed_y - (sep - sty_thick*numstys);
  shift_y_lead = shift_y_plastic - tray_thick;

  SSD = 184.5 - (sty_thick*numstys) - (bed_y - t0_y);

  cout << "shift_y_plastic is " << shift_y_plastic << " cm." << endl;
  cout << "shift_y_lead is " << shift_y_lead << " cm." << endl;
  cout << "SSD is " << SSD << " cm." << endl;

  return 0;

}
