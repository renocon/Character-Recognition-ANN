#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#define MAX 25
#define inMag 24
#define mlaMag 16
#define mlbMag 8
#define outMag 4

#define theta 0
#define iter 400
double alpha = 0.607;
int specs;
int learned = 0;
int itera = 0;
double input[inMag], mla[mlaMag], mlaw[inMag][mlaMag],mlb[mlbMag], mlbw[mlaMag][mlbMag],mlow[mlbMag][outMag],output[outMag],aim[outMag];
FILE * in = fopen("crint.txt","r");

int getDec(double a[]){
    //int len = sizeof(a)/sizeof(int);
    int pow = 1;
    int num = 0;
    for(int x = outMag-1;x>=0;x--){
            int y;
            if(a[x]> 0.5) y = 1;
            else y = 0;
            num+=(y*pow);
            pow*=2;
            }
    return num;           
}

void readMap(){     
     int x = 0;
     char c = 0;
     while(!feof(in) && x<inMag){
          //printf("%d %d\n",feof(in),x);           
          while (c<'0' || c>'1' && !feof(in)){
                c = fgetc(in);
                //printf("char c = %d\n",c);
                }                  
          input[x] = c - '0'; 
          c = 0; 
          x++;       
     }
     //printf("x = %d\n",x);

     }
     
void readAim(){
     int x = 0;
     char c = 0;
     while(!feof(in) && x<outMag){
          //printf("%d %d\n",feof(in),x);           
          while (c<'0' || c>'1' && !feof(in)){
                c = fgetc(in);
                
                }     
          printf("char c = %c\n",c);                   
          aim[x] = c - '0'; 
          c = 0; 
          x++;       
     }
     }     

double randD(){
       return (double)(((double)rand()/(double)RAND_MAX) - 0.5);
       }
       

void popFlat(double a[], int w){
     for(int x = 0; x<w;x++)a[w] = randD();
     }                 
     
void initWeights(){
     //pop2D(mlaw,inMag,mlaMag);
     for(int x = 0; x<inMag;x++){
          for(int y = 0 ; y<mlaMag;y++){
                  mlaw[x][y] = randD();
                }
           }
              
     //pop2D(mlbw,mlaMag,mlbMag);
     for(int x = 0; x<mlaMag;x++){
          for(int y = 0 ; y<mlbMag;y++){
                  mlbw[x][y] = randD();
                }
           }
           
     //pop2D(mlow,mlbMag,outMag);
     for(int x = 0; x<mlbMag;x++){
          for(int y = 0 ; y<outMag;y++){
                  mlow[x][y] = randD();
                }
           }
     }            

double sigmoid(double x){
       return 1/(1+(1/exp(x)));
       }

void getOut(){
     
     //hidden layer 1
     for(int x = 0; x<mlaMag;x++){// for every node in hidden layer a
          mla[x] = 0;   
          for(int y = 0; y<inMag;y++){//for every input
                 mla[x]+=(input[y] * mlaw[y][x]); 
                }
                mla[x]-=theta;
                mla[x] = sigmoid(mla[x]);
           }
           
     //hidden layer 2
     for(int x = 0; x<mlbMag;x++){// for every node output layer
          mlb[x] = 0;   
          for(int y = 0; y<mlaMag;y++){//for every input in hidden layer b
                 mlb[x]+=(mla[y] * mlbw[y][x]); 
                }
                mlb[x]-=theta;
                mlb[x] = sigmoid(mlb[x]);
           } 
           
     //output
     //printf("out:");
     for(int x = 0; x<outMag;x++){// for every node in hidden layer a
          output[x] = 0;   
          for(int y = 0; y<mlbMag;y++){//for every input
                 output[x]+=(mlb[y] * mlow[y][x]); 
                }
                output[x]-=theta;
                output[x] = sigmoid(output[x]);
                //if(output[x]<0.5) output[x] = 0;
                //else output[x] = 1;
               // printf("%1.0f ",output[x]);
           }
     //printf("\n");         
     }
     
void workBack(){
     //output layer weight correction
     for(int k = 0; k< outMag;k++){
           for(int j = 0; j<mlbMag;j++){
                 double dWjk = alpha * mlb[j] * output[k] * (1 - output[k]) * (aim[k] - output[k]);
                 mlow[j][k]+= dWjk;
                 }
           }
     //hidden layer b correction
     double dj = 0;
     for(int k = 0; k< mlbMag;k++){
           for(int j = 0; j<mlaMag;j++){
                 double dj = 0;  
                 
                 for(int z = 0; z< outMag;z++){
                         double dji = mlow[k][z] * output[z]*(1-output[z]) * (aim[z] - output[z]);
                         dj+=dji;
                         }
                         
                 double dWjk = alpha * mla[j] * mlb[k] * (1 - mlb[k]) * dj;//) * (output[k] - output[k]);
                 mlbw[j][k]+= dWjk;
                 }
           }
     
     //hidden layer a correction  
     for(int k = 0; k< mlaMag;k++){
           for(int j = 0; j<inMag;j++){
                // double dj = 0;  
                 
                 //for(int z = 0; z< mlbMag;z++){
                   //      double dji = mlbw[k][z] * output[z]*(1-output[z]) * mlb[z];
                     //    dj+=dji;
                       //  }
                         
                 double dWjk = alpha * input[j] * mla[k] * (1 - mla[k]) * dj;// * dj;//) * (output[k] - output[k]);
                 mlaw[j][k]+= dWjk;
                 }
           }
         
     }     
     
void train(){
    initWeights();
    while(learned<specs*0.75 && itera++ < iter) {
        learned = 0;      
        in = fopen("crint.txt","r");
        fscanf(in,"%d",&specs);            
        for(int a = 0; a<specs;a++){                  
            int aimDec,resDec = -1;
            readMap();
            readAim();
            aimDec = getDec(aim);
            
            int x = 0;
            while(x++ < iter && aimDec!=resDec){
                 getOut();
                 resDec = getDec(output);
                 printf("aim = %d, output = %d\n",aimDec,resDec);
                 workBack();
            }//end while x++
            if(x==1) learned++;
        }//end for a     
    }//end while learneed 
}//end train     

void read(){
     int x = 0;
     char c = 0;
     while(x<inMag){
          //printf("%d %d\n",feof(in),x);           
          while (c<'0' || c>'1'){
                //c = getc(in);
                scanf("%c",&c);
                }     
          //printf("char c = %c\n",c);                   
          input[x] = c - '0'; 
          c = 0; 
          x++;       
          }
     
     }
     
void save(void);     
void load(){
       FILE * tdata = fopen("train_data.txt","r");
       if(!tdata){
                  printf("No previous training data found. Training now!\n");
                  fscanf(in,"%d",&specs);
                  train();
                  save();
                  return;
                  }
       //mlaw
       for(int x = 0;x<inMag;x++){
           for(int y = 0; y<mlaMag;y++){
                   fscanf(tdata,"%lf", &mlaw[x][y]);
                 }    
             }
       //mlbw
       for(int x = 0;x<mlaMag;x++){
           for(int y = 0; y<mlbMag;y++){
                   fscanf(tdata,"%lf", &mlbw[x][y]);
                 }    
             }
       //mlow
       
       for(int x = 0;x<mlbMag;x++){
           for(int y = 0; y<outMag;y++){
                   fscanf(tdata,"%lf", &mlow[x][y]);
                 }    
             }
             
       fscanf(tdata,"%lf ",&alpha);      
       fclose(tdata);
       }
       
void save(){
       FILE * out = fopen("train_data.txt","w");
       //mlaw
       for(int x = 0;x<inMag;x++){
           for(int y = 0; y<mlaMag;y++){
                   fprintf(out,"%lf ", mlaw[x][y]);
                 }    
             }
       //mlbw
       for(int x = 0;x<mlaMag;x++){
           for(int y = 0; y<mlbMag;y++){
                   fprintf(out,"%lf ", mlbw[x][y]);
                 }    
             }
       //mlow
       
       for(int x = 0;x<mlbMag;x++){
           for(int y = 0; y<outMag;y++){
                   fprintf(out,"%lf ", mlow[x][y]);
                 }    
             }
       fprintf(out,"%lf ",alpha);
       fclose(out);
       }            

void readFile(){
     printf("Reading file...\n");
     FILE * inp = fopen("input.txt","r");
     if(!inp){
              printf("no input file!\n");
              system("pause");
              exit(1);
              }
     int x = 0;
     char c = 0;
     while(x<inMag){
          //printf("%d %d\n",feof(in),x);           
          while (c<'0' || c>'1'){
                //c = getc(in);
                fscanf(inp,"%c",&c);
                }     
          //printf("char c = %c\n",c);                   
          input[x] = c - '0'; 
          c = 0; 
          x++;       
          }
     fclose(inp);     
     }

int main(){
    
    //printf("specs: %d",specs);
    srand((unsigned)time(NULL));
    int resp = 0;
    printf("Do Training?\n1) Yes\n2) No\n");
    scanf("%d",&resp);
    if(resp == 1){
        //alpha = randD();  
        //if(alpha<0.1) alpha = 0.607061 ;  
        if(!in){
                printf("no training file\n");
                system("pause");
            return 1;
            }
        fscanf(in,"%d",&specs);
        train();
        //clrscr();
        //clc();
        save();
        printf("Training Complete!\n");
        
        }
    else{
         load();
         }    
    //printf("input a number matrix of size %d\n",inMag);
    //read();
    readFile();
    getOut();
    printf("Result: %d\n", getDec(output));
    system("pause");
    return 0;    
}

