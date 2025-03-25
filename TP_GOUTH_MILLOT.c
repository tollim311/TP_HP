#include <stdio.h>
#include <stdlib.h>

int main(void){

    //Input File name
    FILE *fIn = fopen("lena.bmp","r");
    //Output File name
	FILE *fOut = fopen("out.bmp","w+");

    // Goes to end of file and gets the size of the file
    fseek(fIn, 0, SEEK_END);
    int size_img = ftell(fIn);
    rewind(fIn);

    //to store the image data
    unsigned char buffer[size_img];

    //read image data
	fread(buffer,sizeof(unsigned char),size_img,fIn);
	
    //write image data to output file
	fwrite(buffer,sizeof(unsigned char),size_img,fOut);

    fclose(fIn);
	fclose(fOut);

	return 0;
}