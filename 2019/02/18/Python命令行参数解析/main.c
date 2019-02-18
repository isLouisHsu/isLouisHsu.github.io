#include<stdio.h>

int main(int argc,char * argv[],char * envp[])
{
    printf("argc is %d \n", argc);
 
    int i;
    for (i=0; i<argc; i++)
    {
        printf("arcv[%d] is %s\n", i, argv[i]);
    }

    for (i=0; envp[i]!=NULL; i++)
    {
        printf("envp[%d] is %s\n", i, envp[i]);
    }
	
	return 0;
}
