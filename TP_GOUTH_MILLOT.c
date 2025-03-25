#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

int main(void){

    FILE *fIn = fopen("lena.bmp","r");				//Input File name
	FILE *fOut = fopen("out.bmp","w+");		            //Output File name

	int i;
	unsigned char byte[54];								//to get the image header
	unsigned char colorTable[1024];						//to get the colortable

	if(fIn==NULL)										// check if the input file has not been opened succesfully.
	{										
		printf("File does not exist.\n");
	}

	for(i=0;i<54;i++)									//read the 54 byte header from fIn
	{									
		byte[i]=getc(fIn);								
	}

	fwrite(byte,sizeof(unsigned char),54,fOut);			//write the header back

	// extract image height, width and bitDepth from imageHeader 
	int height = *(int*)&byte[18];
	int width = *(int*)&byte[22];
	int bitDepth = *(int*)&byte[28];

	int size=height*width;								//calculate image size

	unsigned char buffer[size];							//to store the image data

	fread(buffer,sizeof(unsigned char),size,fIn);		//read image data
	
	fwrite(buffer,sizeof(unsigned char),size,fOut);		//write back to the output image

	fclose(fIn);
	fclose(fOut);

	return 0;
}