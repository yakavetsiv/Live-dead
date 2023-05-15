dir = getDirectory ("image");

prefix="D4-132_d8_";

n0=1;
n=12;

x_offset = 3150;
y_offest = -5;

Roi.getBounds(x0,y0,w0,h0);

for (i=1; i<=n; i+=1){ 
	makeRectangle(x0+x_offset*(i-1), y0+y_offest*(i-1), w0, h0);
	run("Duplicate...", "duplicate");
	number = n0 + i -1;
	saveAs("Tiff", dir + prefix + number + ".tif");
	close();
}


//run("Duplicate...", "duplicate");
//saveAs("Tiff", "F:/Projects/MIcrofluidics-liposomes/data/Microscopy/2023/Mar 8, 2023 - D4-115 Lip 100 MCF-7/D4-115_48h_3.tif");
//selectWindow("D4-115_48h_fl.nd2");
//makeRectangle(3312, 976, 2128, 8544);
