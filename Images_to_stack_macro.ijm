selectImage(1);
run("Duplicate...", " ");
run("Invert");
run("Brightness/Contrast...");
setMinAndMax(15000, 16383);
run("Apply LUT");
run("Images to Stack", "name=Stack title=1");
stack = getImageID();
run("Make Substack...", "slices=3,1,2");
selectImage(stack);
close();




