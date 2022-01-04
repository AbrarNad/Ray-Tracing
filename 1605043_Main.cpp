#include<cmath>
#include<bits/stdc++.h>
#include<iostream>
#include<fstream>
#include <time.h>
#include <windows.h>
#include <GL/glut.h>
#include "bitmap_image.hpp"

#define pi (2*acos(0.0))
#define bulletNum 50

using namespace std;

double cameraHeight;
double cameraAngle;
int drawgrid;
int drawaxes;
double angle;
double rotate1=0,rotate2=0,rotate3=0,rotate4=0,rotate5=0,rotate6=0,rotate7=0,rotate8=0;

struct point
{
	double x,y,z;
};

struct bullet{
    double d1, d2, d3, d4;
};

struct point multVectScalar(struct point p, double a)
{
	struct point ret;

	ret.x=p.x*a;
	ret.y=p.y*a;
	ret.z=p.z*a;

	return ret;
};

struct point sumVect(struct point p,struct point q)
{
	struct point ret;

	ret.x=p.x+q.x;
	ret.y=p.y+q.y;
	ret.z=p.z+q.z;

	return ret;
};

double dotMult(point v, point w){
    double ret;
    ret = v.x*w.x+v.y*w.y+v.z*w.z;

    return ret;
};

struct point crossMult(struct point v,struct point w)
{
    struct point ret;

    ret.x = v.y*w.z - v.z*w.y;
    ret.y = v.z*w.x - v.x*w.z;
    ret.z = v.x*w.y - v.y*w.x;

    return ret;
};

struct point getUnitVect(struct point p){
    struct point ret;

    double s = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
    ret.x = p.x/s;
    ret.y = p.y/s;
    ret.z = p.z/s;

    return ret;
};

struct point rotateAroundAxis(struct point v, struct point k, double a)
{
    struct point ret;

    ret = multVectScalar(v,cos(a*pi/180.0));
    ret = sumVect(ret,multVectScalar(crossMult(k,v),sin(a*pi/180.0)));

    return ret;
};

double getVectorModulus(struct point p){
    double ret;
    ret = dotMult(p,p);
    ret = sqrt(ret);

    return ret;
}

struct point Rotate(point x, point a, double angle){
    struct point ret,tmp;

    ret = multVectScalar(x,cos(angle*pi/180.0));
    ret = sumVect(ret,multVectScalar(crossMult(a,x),sin(angle*pi/180.0)));
    tmp = multVectScalar(a,1-cos(angle*pi/180.0));
    tmp = multVectScalar(tmp,dotMult(a,x));
    ret = sumVect(ret,tmp);

    return ret;
};

void printPoint(struct point p)
{
    cout<<endl<<endl;
    cout<<p.x<<" "<<p.y<<" "<<p.z<<endl;
}

void drawSphere(double radius,int slices,int stacks, double color_r, double color_g, double color_b)
{
	struct point points[100][100];
	int i,j;
	double h,r;
	//generate points
	for(i=0;i<=stacks;i++)
	{
		h=radius*sin(((double)i/(double)stacks)*(pi/2));
		r=radius*cos(((double)i/(double)stacks)*(pi/2));
		for(j=0;j<=slices;j++)
		{
			points[i][j].x=r*cos(((double)j/(double)slices)*2*pi);
			points[i][j].y=r*sin(((double)j/(double)slices)*2*pi);
			points[i][j].z=h;
		}
	}
	//draw quads using generated points
	for(i=0;i<stacks;i++)
	{
        glColor3f(color_r,color_g,color_b);
		for(j=0;j<slices;j++)
		{
			glBegin(GL_QUADS);{
			    //upper hemisphere
				glVertex3f(points[i][j].x,points[i][j].y,points[i][j].z);
				glVertex3f(points[i][j+1].x,points[i][j+1].y,points[i][j+1].z);
				glVertex3f(points[i+1][j+1].x,points[i+1][j+1].y,points[i+1][j+1].z);
				glVertex3f(points[i+1][j].x,points[i+1][j].y,points[i+1][j].z);
                //lower hemisphere
                glVertex3f(points[i][j].x,points[i][j].y,-points[i][j].z);
				glVertex3f(points[i][j+1].x,points[i][j+1].y,-points[i][j+1].z);
				glVertex3f(points[i+1][j+1].x,points[i+1][j+1].y,-points[i+1][j+1].z);
				glVertex3f(points[i+1][j].x,points[i+1][j].y,-points[i+1][j].z);
			}glEnd();
		}
	}
}

struct point pos, up, rightp, look;

class Ray {
public:
    struct point start, dir;

    Ray(struct point start, struct point direction){
        this->start = start;
        this->dir = direction;
    }
};

class Object{
public:

    struct point ref_point;
    double height, width, length;
    double color[3];
    double coEfficients[4];
    int shine;

    Object(){
    }

    virtual void draw(){}
    virtual void print(){}
    virtual double getIntersectingT(Ray r){}
    virtual double intersect(Ray r, double *color, int level){}

    void setColor(int r, int g, int b){
        this->color[0] = r;
        this->color[1] = g;
        this->color[2] = b;
    }

    void setShine(int shine){
        this->shine = shine;
    }

    void setCoEfficients(double amb, double diff, double spec, double reflect){
        this->coEfficients[0] = amb;
        this->coEfficients[1] = diff;
        this->coEfficients[2] = spec;
        this->coEfficients[3] = reflect;
    }
};


class Light{
public:
    struct point light_pos;
    double color[3];

    Light(struct point pos, double color_r, double color_g, double color_b){
        this->light_pos = pos;
        this->color[0] = color_r;
        this->color[1] = color_g;
        this->color[2] = color_b;
    }

    void draw() {
		glPushMatrix();
		glTranslated(this->light_pos.x, this->light_pos.y, this->light_pos.z);
		drawSphere(2, 8, 8, this->color[0], this->color[1], this->color[2]);
		glPopMatrix();
	}

    void print(){
        cout<<"\t**light**"<<endl;
        cout<<"pos: "<<this->light_pos.x<<" "<<this->light_pos.y<<" "<<this->light_pos.z<<endl;
        cout<<"color: "<<this->color[0]<<" "<<this->color[1]<<" "<<this->color[2]<<endl;
    }
};

///for offline 3

vector<Object*> objects;
vector<Light> lights;
int recursionDepth, dimensions;

class Sphere : public Object{
public:
    double radius;

    Sphere(double x, double y, double z, double rad){
        this->ref_point.x = x;
        this->ref_point.y = y;
        this->ref_point.z = z;
        this->radius  = rad;
    }

    void print(){
        cout<<"\t**sphere**"<<endl;
        cout<<"Center: "<< ref_point.x<<" "<<ref_point.y<<" "<<ref_point.z<<endl;
        cout<<"radius: "<<radius<<endl;
        cout<<"color: "<< this->color[0]<<" "<<this->color[1]<<" "<<this->color[2]<<endl;
        cout<<"coeffs: "<<this->coEfficients[0]<<" "<<this->coEfficients[1]<<" "<<this->coEfficients[2]<<" "<<this->coEfficients[3]<<endl;
        cout<<"shine: "<< this->shine<<endl<<endl;
    }

    void draw()
    {
        glPushMatrix();
		glTranslated(this->ref_point.x, this->ref_point.y, this->ref_point.z);
		drawSphere(radius, 30, 30, this->color[0], this->color[1], this->color[2]);
		glPopMatrix();
    }

    double getIntersectingT(Ray r){
        struct point center = this->ref_point;
        double t;

        struct point r0 = sumVect(r.start, multVectScalar(center,-1));
        struct point rd = r.dir;

        double a = 1;
        double b = dotMult(r0,rd)*2;
        double c = dotMult(r0,r0) - radius*radius;
        double d = b*b - a*c*4;

        if(d<0){
            return -1;
        }
        double alpha = (-b + sqrt(d))/2;
        double beta = (-b - sqrt(d))/2;

        if(alpha>0){
            if(beta<0) t = alpha;
            else t = min(alpha, beta);
        }else{
            if(beta>0) t = beta;
            else t = -1;
        }

        return t;
    }

    double intersect(Ray ray, double *color, int level){
        double t = getIntersectingT(ray);

        if(t<0) return -1;
        if(level == 0) return t;
        else{
            struct point intersectionPoint = sumVect(ray.start, multVectScalar(ray.dir,t));
            //struct point intersectionPoint = sumVect(sumVect(ray.start,multVectScalar(ref_point,-1)), multVectScalar(ray.dir,t));
            struct point normal = sumVect(intersectionPoint, multVectScalar(this->ref_point,-1));
            //struct point normal = sumVect(this->ref_point, multVectScalar(intersectionPoint,-1));
            normal = getUnitVect(normal);
            point reflect = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));///dir - 2*n*(n.dir)
            reflect = multVectScalar(reflect,-1);
            reflect = getUnitVect(reflect);

            color[0] = (this->color[0])*(this->coEfficients[0]);
            color[1] = (this->color[1])*(this->coEfficients[0]);
            color[2] = (this->color[2])*(this->coEfficients[0]);

            for(int i=0; i<lights.size(); i++){
                point lightdir = sumVect(lights[i].light_pos, multVectScalar(intersectionPoint,-1));
                //point lightdir = sumVect(intersectionPoint, multVectScalar(lights[i].light_pos,-1));
                double rayL = getVectorModulus(lightdir);
                lightdir = getUnitVect(lightdir);

                point raystart = sumVect(intersectionPoint,multVectScalar(lightdir,1));
                point raydir = sumVect(lights[i].light_pos,multVectScalar(intersectionPoint,-1));
                raydir = getUnitVect(raydir);

                Ray rayLight(raystart,raydir);

                bool inShadow = false;

                for(int j=0; j<objects.size(); j++){
                    double t = objects[j]->getIntersectingT(rayLight);
                    if(t>0 && t<rayL){
                        inShadow = true;
                        break;
                    }
                }

                if(inShadow==0){
                    double lamb = dotMult(rayLight.dir,normal);///L.N
                    point reflectPhong = sumVect(rayLight.dir,multVectScalar(normal,-2*dotMult(rayLight.dir,normal)));
                    reflectPhong = getUnitVect(reflectPhong);
                    double val = (dotMult(multVectScalar(ray.dir,-1),reflectPhong));///R.V
                    double phong = pow(max(val,0.0),shine);///R.V^k
                    lamb = max(lamb,0.0);
                    //phong = phong;

                    color[0]+= this->color[0]*lights[i].color[0]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[1]+= this->color[1]*lights[i].color[1]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[2]+= this->color[2]*lights[i].color[2]*(lamb*coEfficients[1]+phong*coEfficients[2]);

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            if(level >= recursionDepth) return t;

            {
                point reflectRecur = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));
                reflectRecur = getUnitVect(reflectRecur);
                point start = sumVect(intersectionPoint,multVectScalar(reflectRecur,1));
                Ray reflectionRay(start,reflectRecur);
                double *reflectedColor = new double[3];
                double tnew, tmin=99999;
                int nearest = -1;

                for(int i=0;i<objects.size();i++){
                    tnew = objects[i]->intersect(reflectionRay,reflectedColor,0);
                    if(tnew>0){
                        if(tnew<tmin){
                            nearest = i;
                            tmin = tnew;
                        }
                    }
                }

                if(nearest != -1){
                    objects[nearest]->intersect(reflectionRay,reflectedColor,level+1);
                    //if(level==recursionDepth){
                        color[0]+=reflectedColor[0]*coEfficients[3];
                        color[1]+=reflectedColor[1]*coEfficients[3];
                        color[2]+=reflectedColor[2]*coEfficients[3];
                    //}

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            for(int k=0; k<3;k++){
                if(color[k]<0) color[k] = 0;
                if(color[k]>1) color[k] = 1;
            }

            return t;
        }
    }
};

class Triangle : public Object {
public:
    struct point a,b,c;

    Triangle(point a, point b, point c){
        this->a = a;
        this->b = b;
        this->c = c;
    }

    void draw(){
        glBegin(GL_TRIANGLES);
        glColor3f(this->color[0], this->color[1], this->color[2]);
        glVertex3f(this->a.x, this->a.y, this->a.z);
        glVertex3f(this->b.x, this->b.y, this->b.z);
        glVertex3f(this->c.x, this->c.y, this->c.z);
        glEnd();
    }

    void print(){
        cout<<"\t**triangle**"<<endl;
        cout<<"point a: "<<this->a.x<<" "<<this->a.y<<" "<<this->a.z<<endl;
        cout<<"point b: "<<this->b.x<<" "<<this->b.y<<" "<<this->b.z<<endl;
        cout<<"point c: "<<this->c.x<<" "<<this->c.y<<" "<<this->c.z<<endl;
        cout<<"color: "<< this->color[0]<<" "<<this->color[1]<<" "<<this->color[2]<<endl;
        cout<<"coeffs: "<<this->coEfficients[0]<<" "<<this->coEfficients[1]<<" "<<this->coEfficients[2]<<" "<<this->coEfficients[3]<<endl;
        cout<<"shine: "<< this->shine<<endl;
    }

    double getIntersectingT(Ray ray){
        point s,q;
        point edge1 = sumVect(this->b,multVectScalar(this->a,-1)); ///b-a
        point edge2 = sumVect(this->c,multVectScalar(this->a,-1));  ///c-a
        point normal = crossMult(ray.dir,edge2);
        float t,a,f,u,v;

        a = dotMult(edge1,normal);
        if(a>-0.0000001 && a<0.0000001){
            return -1;
        }

        f= 1.0/a;
        s = sumVect(ray.start,multVectScalar(this->a,-1));
        u = f*dotMult(s,normal);
        if(u<0.0 || u >1.0 ) return -1;

        q = crossMult(s,edge1);
        v = f*dotMult(q,ray.dir);
        if(v<0.0 || u+v > 1.0)  return -1;

        ///calculate t
        t = f*dotMult(q,edge2);

        if(t>0.0000001){
            return t;
        }

        return -1;
    }

    double intersect(Ray ray, double *color, int level){
        double t = getIntersectingT(ray);

        if(t<0) return -1;
        if(level == 0)  return t;
        else{
            struct point intersectionPoint = sumVect(ray.start, multVectScalar(ray.dir,t));

            ///b-a*c-a
            point edge1 = sumVect(this->b,multVectScalar(this->a,-1));
            point edge2 = sumVect(this->c,multVectScalar(this->a,-1));

            point normal = crossMult(edge1, edge2);
            normal = getUnitVect(normal);
            point reflect = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));///dir - 2*n*(n.dir)
            reflect = getUnitVect(reflect);

            color[0] = (this->color[0])*(this->coEfficients[0]);
            color[1] = (this->color[1])*(this->coEfficients[0]);
            color[2] = (this->color[2])*(this->coEfficients[0]);

            for(int i=0; i<lights.size(); i++){
                point lightdir = sumVect(lights[i].light_pos, multVectScalar(intersectionPoint,-1));
                //point lightdir = sumVect(intersectionPoint, multVectScalar(lights[i].light_pos,-1));
                double rayL = getVectorModulus(lightdir);
                lightdir = getUnitVect(lightdir);

                point raystart = sumVect(intersectionPoint,multVectScalar(lightdir,1));
                point raydir = sumVect(lights[i].light_pos,multVectScalar(intersectionPoint,-1));
                raydir = getUnitVect(raydir);

                Ray rayLight(raystart,raydir);

                bool inShadow = false;

                for(int j=0; j<objects.size(); j++){
                    double t = objects[j]->getIntersectingT(rayLight);
                    if(t>0 && t<rayL){
                        inShadow = true;
                        break;
                    }
                }

                if(inShadow==0){
                    double lamb = dotMult(rayLight.dir,normal);///L.N
                    point reflectPhong = sumVect(rayLight.dir,multVectScalar(normal,-2*dotMult(rayLight.dir,normal)));
                    reflectPhong = getUnitVect(reflectPhong);
                    double val = (dotMult(multVectScalar(ray.dir,-1),reflectPhong));///R.V
                    double phong = pow(max(val,0.0),shine);///R.V^k
                    lamb = max(lamb,0.0);
                    //phong = phong;

                    color[0]+= this->color[0]*lights[i].color[0]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[1]+= this->color[1]*lights[i].color[1]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[2]+= this->color[2]*lights[i].color[2]*(lamb*coEfficients[1]+phong*coEfficients[2]);

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            if(level >= recursionDepth) return t;

            {
                point reflectRecur = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));
                reflectRecur = getUnitVect(reflectRecur);
                point start = sumVect(intersectionPoint,multVectScalar(reflectRecur,1));
                Ray reflectionRay(start,reflectRecur);
                double *reflectedColor = new double[3];
                double tnew, tmin=99999;
                int nearest = -1;

                for(int i=0;i<objects.size();i++){
                    tnew = objects[i]->intersect(reflectionRay,reflectedColor,0);
                    if(tnew>0){
                        if(tnew<tmin){
                            nearest = i;
                            tmin = tnew;
                        }
                    }
                }

                if(nearest != -1){
                    objects[nearest]->intersect(reflectionRay,reflectedColor,level+1);
                    //if(level==recursionDepth){
                        color[0]+=reflectedColor[0]*coEfficients[3];
                        color[1]+=reflectedColor[1]*coEfficients[3];
                        color[2]+=reflectedColor[2]*coEfficients[3];
                    //}

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            for(int k=0; k<3;k++){
                if(color[k]<0) color[k] = 0;
                if(color[k]>1) color[k] = 1;
            }

            return t;
        }
    }
};

class General : public Object {
public:
    double a, b, c, d, e, f, g, h, i, j;

    General(double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double length, double width, double height){
        this-> a = a;
        this-> b = b;
        this-> c = c;
        this-> d = d;
        this-> e = e;
        this-> f = f;
        this-> g = g;
        this-> h = h;
        this-> i = i;
        this-> j = j;
        this-> length = length;
        this-> width = width;
        this-> height = height;
    }

    void draw(){
        ///nothing
    }

    void print(){
        cout<<"\t**general**"<<endl;
        cout<<"eqn: "<<this->a<<" "<<this->b<<" "<<this->c<<" "<<this->d<<" "<<this->e<<" "<<this->f<<" "<<this->g<<" "<<this->h<<" "<<this->i<<" "<<this->j<<endl;
        cout<<"length w h: "<< this->length<<" "<< this->width<<" "<< this->height<<endl;
        cout<<"point ref: "<<this->ref_point.x<<" "<<this->ref_point.y<<" "<<this->ref_point.z<<endl;
        cout<<"color: "<< this->color[0]<<" "<<this->color[1]<<" "<<this->color[2]<<endl;
        cout<<"coeffs: "<<this->coEfficients[0]<<" "<<this->coEfficients[1]<<" "<<this->coEfficients[2]<<" "<<this->coEfficients[3]<<endl;
        cout<<"shine: "<< this->shine<<endl;
    }

    double getIntersectingT(Ray ray){
        struct point intersecting;
        double t;
        double Aq, Bq, Cq;
        Aq = this->a*(ray.dir.x*ray.dir.x) + this->b*(ray.dir.y*ray.dir.y) + this->c*(ray.dir.z*ray.dir.z) + this->d*(ray.dir.x*ray.dir.y) + this->e*(ray.dir.x*ray.dir.z) + this->f*(ray.dir.y*ray.dir.z);

        Bq = 2*a*ray.dir.x*ray.start.x + 2*b*ray.dir.y*ray.start.y + 2*c*ray.dir.z*ray.start.z + d*ray.dir.y*ray.start.x + d*ray.start.y*ray.dir.x + e*ray.start.x*ray.dir.z + e*ray.start.z*ray.dir.x + f*ray.start.y*ray.dir.z + f*ray.start.z*ray.dir.y + g*ray.dir.x + h*ray.dir.y + i*ray.dir.z;

        Cq = a*ray.start.x*ray.start.x + b* ray.start.y*ray.start.y + c*ray.start.z*ray.start.z + d*ray.start.x*ray.start.y + e*ray.start.x*ray.start.z + f*ray.start.y*ray.start.z + g*ray.start.x + h*ray.start.y + i*ray.start.z + j;

        if( Aq == 0){
            t = -Cq/Bq;
            if(t<0) return -1;
            intersecting = sumVect(ray.start, multVectScalar(ray.dir,t));
            if(intersecting.x>ref_point.x+length || intersecting.y>ref_point.y+width || intersecting.z>ref_point.z+height)  return -1;
            else    return t;
        }


        double deter = Bq*Bq - 4*Aq*Cq;
        if(deter < 0)  return -1;

        double t1 = (-Bq - pow(deter,0.5))/(2*Aq);
        double t2 = (-Bq + pow(deter,0.5))/(2*Aq);


        if(t1>0){
            if(t2<0){
                t = t1;
                intersecting = sumVect(ray.start, multVectScalar(ray.dir,t));
                if(length != 0){
                    if(intersecting.x < ref_point.x || intersecting.x>ref_point.x+length)   return -1;
                }
                if(width != 0){
                    if(intersecting.y < ref_point.y || intersecting.y>ref_point.y+width)   return -1;
                }
                if(height != 0){
                    if(intersecting.z < ref_point.z || intersecting.z>ref_point.z+height)   return -1;
                }
                return t;
            }else{
                ///both positive
                bool flag1 = true, flag2 = true;
                point intersecting1 = sumVect(ray.start, multVectScalar(ray.dir,t1));
                if(length != 0){
                    if(intersecting1.x < ref_point.x || intersecting1.x>ref_point.x+length)   flag1 = false;
                }
                if(width != 0){
                    if(intersecting1.y < ref_point.y || intersecting1.y>ref_point.y+width)   flag1 = false;
                }
                if(height != 0){
                    if(intersecting1.z < ref_point.z || intersecting1.z>ref_point.z+height)   flag1 = false;
                }

                point intersecting2 = sumVect(ray.start, multVectScalar(ray.dir,t2));

                if(length != 0){
                    if(intersecting2.x < ref_point.x || intersecting2.x>ref_point.x+length)   flag2 = false;
                }
                if(width != 0){
                    if(intersecting2.y < ref_point.y || intersecting2.y>ref_point.y+width)   flag2 = false;
                }
                if(height != 0){
                    if(intersecting2.z < ref_point.z || intersecting2.z>ref_point.z+height)   flag2 = false;
                }

                if(flag1 && !flag2) return t1;
                if(!flag1 && flag2) return t2;
                if(flag1 && flag2)  return min(t1,t2);
                if(!flag1 && !flag2)    return -1;
            }
        }
        if(t2>0){       ///t1 neg t2 pos
                t = t2;
                intersecting = sumVect(ray.start, multVectScalar(ray.dir,t));
                if(length != 0){
                    if(intersecting.x < ref_point.x || intersecting.x>ref_point.x+length)   return -1;
                }
                if(width != 0){
                    if(intersecting.y < ref_point.y || intersecting.y>ref_point.y+width)   return -1;
                }
                if(height != 0){
                    if(intersecting.z < ref_point.z || intersecting.z>ref_point.z+height)   return -1;
                }
                return t;
        }
        ///both neg
        return -1;
    }

    double intersect(Ray ray, double *color, int level){
        struct point normal;
        double t = getIntersectingT(ray);

        //cout<<t<<endl;
        if(t<0)
            return -1;
        else if(level == 0)
            return t;
        else{
            //cout<<"dangerous jayga"<<endl;

            struct point intersectionPoint = sumVect(ray.start, multVectScalar(ray.dir,t));

            normal.x = 2*a*intersectionPoint.x + d*intersectionPoint.y + e*intersectionPoint.z + g;
            normal.y = 2*b*intersectionPoint.y + d*intersectionPoint.x + f*intersectionPoint.z + h;
            normal.z = 2*c*intersectionPoint.z + e*intersectionPoint.x + f*intersectionPoint.y + i;
            point reflect = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));///dir - 2*n*(n.dir)
            reflect = getUnitVect(reflect);
            normal = getUnitVect(normal);

            color[0] = (this->color[0])*(this->coEfficients[0]);
            color[1] = (this->color[1])*(this->coEfficients[0]);
            color[2] = (this->color[2])*(this->coEfficients[0]);

            for(int i=0; i<lights.size(); i++){
                point lightdir = sumVect(lights[i].light_pos, multVectScalar(intersectionPoint,-1));
                //point lightdir = sumVect(intersectionPoint, multVectScalar(lights[i].light_pos,-1));
                double rayL = getVectorModulus(lightdir);
                lightdir = getUnitVect(lightdir);

                point raystart = sumVect(intersectionPoint,multVectScalar(lightdir,1));
                point raydir = sumVect(lights[i].light_pos,multVectScalar(intersectionPoint,-1));
                raydir = getUnitVect(raydir);

                Ray rayLight(raystart,raydir);

                bool inShadow = false;

                for(int j=0; j<objects.size(); j++){
                    double t = objects[j]->getIntersectingT(rayLight);
                    if(t>0 && t<rayL){
                        inShadow = true;
                        break;
                    }
                }

                if(inShadow==0){
                    double lamb = dotMult(rayLight.dir,normal);///L.N
                    point reflectPhong = sumVect(rayLight.dir,multVectScalar(normal,-2*dotMult(rayLight.dir,normal)));
                    reflectPhong = getUnitVect(reflectPhong);
                    double val = (dotMult(multVectScalar(ray.dir,-1),reflectPhong));///R.V
                    double phong = pow(max(val,0.0),shine);///R.V^k
                    lamb = max(lamb,0.0);
                    //phong = phong;

                    color[0]+= this->color[0]*lights[i].color[0]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[1]+= this->color[1]*lights[i].color[1]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[2]+= this->color[2]*lights[i].color[2]*(lamb*coEfficients[1]+phong*coEfficients[2]);

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            if(level >= recursionDepth) return t;

            {
                point reflectRecur = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));
                reflectRecur = getUnitVect(reflectRecur);
                point start = sumVect(intersectionPoint,multVectScalar(reflectRecur,1));
                Ray reflectionRay(start,reflectRecur);
                double *reflectedColor = new double[3];
                double tnew, tmin=99999;
                int nearest = -1;

                for(int i=0;i<objects.size();i++){
                    tnew = objects[i]->intersect(reflectionRay,reflectedColor,0);
                    if(tnew>0){
                        if(tnew<tmin){
                            nearest = i;
                            tmin = tnew;
                        }
                    }
                }

                if(nearest != -1){
                    objects[nearest]->intersect(reflectionRay,reflectedColor,level+1);
                    //if(level==recursionDepth){
                        color[0]+=reflectedColor[0]*coEfficients[3];
                        color[1]+=reflectedColor[1]*coEfficients[3];
                        color[2]+=reflectedColor[2]*coEfficients[3];
                    //}

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            for(int k=0; k<3;k++){
                if(color[k]<0) color[k] = 0;
                if(color[k]>1) color[k] = 1;
            }

            return t;
        }
    }

};

class Floor : public Object {
public:
    int floorWidth, tileWidth;

    Floor(int floorWidth, int tileWidth){
        this->floorWidth = floorWidth;
        this->tileWidth = tileWidth;
        this->ref_point.x = -floorWidth/2;
        this->ref_point.y = -floorWidth/2;
        this->ref_point.z = 0;
        this->coEfficients[0] = .4;
        this->coEfficients[1] = .2;
        this->coEfficients[2] = .3;
        this->coEfficients[3] = .3;
        this->shine = 4;
    }

    void draw(){
        glBegin(GL_QUADS);
        {
            int nSquares = floorWidth/tileWidth;
            for (int i = 0; i<nSquares; i++) {
                for (int j = 0; j<nSquares; j++) {
                    if ((i+j)%2 == 0)
                        glColor3f(1, 1, 1);
                    else
                        glColor3f(0, 0, 0);
                    glVertex3f(ref_point.x + i*tileWidth,ref_point.y +  j*tileWidth, 0);
                    glVertex3f(ref_point.x + i*tileWidth + tileWidth,ref_point.y +  j*tileWidth, 0);
                    glVertex3f(ref_point.x + i*tileWidth + tileWidth,ref_point.y +  j*tileWidth + tileWidth, 0);
                    glVertex3f(ref_point.x + i*tileWidth,ref_point.y +  j*tileWidth + tileWidth, 0);
                }
            }
        }
        glEnd();
    }

    void getColorAtPoint(struct point ip, double *color){
        double dx = ip.x-this->ref_point.x;
        double dy = ip.y-this->ref_point.y;
        int i = (int)(dx/this->tileWidth);
        int j = (int)(dy/this->tileWidth);

        if((i+j)%2==0){
            color[0]=1;
            color[1]=1;
            color[2]=1;
        }else{
            color[0]=0;
            color[1]=0;
            color[2]=0;
        }
    }

    double getIntersectingT(Ray r){
        struct point normal;
        normal.x = 0;
        normal.y = 0;
        normal.z = 1;
        if(r.dir.z == 0)    return -1;      ///corner case
        double t = -(dotMult(r.start,normal)/dotMult(r.dir,normal));
        return t;
    }

    double intersect(Ray ray, double *color, int level){
        struct point normal;
        normal.x = 0;
        normal.y = 0;
        normal.z = 1;
        double t;
        if(dotMult(ray.dir,normal)==0)  return -1;
        t = getIntersectingT(ray);
        //cout<<t<<endl;
        if(t<0)
            return -1;
        else if(level == 0)
            return t;
        else{
            //cout<<"dangerous jayga"<<endl;
            struct point intersectionPoint = sumVect(ray.start, multVectScalar(ray.dir,t));
            if(intersectionPoint.x > floorWidth/2 || intersectionPoint.x < -floorWidth/2)
                return -1;
            if(intersectionPoint.y > floorWidth/2 || intersectionPoint.y < -floorWidth/2)
                return -1;
            getColorAtPoint(intersectionPoint,color);
            double tempColor[3];

            tempColor[0] = (color[0]);
            tempColor[1] = (color[1]);
            tempColor[2] = (color[2]);
            color[0] = (color[0])*(this->coEfficients[0]);
            color[1] = (color[1])*(this->coEfficients[0]);
            color[2] = (color[2])*(this->coEfficients[0]);

            point reflect = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));///dir - 2*n*(n.dir)
            reflect = getUnitVect(reflect);

            for(int i=0; i<lights.size(); i++){
                point lightdir = sumVect(lights[i].light_pos, multVectScalar(intersectionPoint,-1));
                //point lightdir = sumVect(intersectionPoint, multVectScalar(lights[i].light_pos,-1));
                double rayL = getVectorModulus(lightdir);
                lightdir = getUnitVect(lightdir);

                point raystart = sumVect(intersectionPoint,multVectScalar(lightdir,1));
                point raydir = sumVect(lights[i].light_pos,multVectScalar(intersectionPoint,-1));
                raydir = getUnitVect(raydir);

                Ray rayLight(raystart,raydir);

                bool inShadow = false;

                for(int j=0; j<objects.size(); j++){
                    double t = objects[j]->getIntersectingT(rayLight);
                    if(t>0 && t<rayL){
                        inShadow = true;
                        break;
                    }
                }

                if(inShadow==0){
                    double lamb = dotMult(rayLight.dir,normal);///L.N
                    point reflectPhong = sumVect(rayLight.dir,multVectScalar(normal,-2*dotMult(rayLight.dir,normal)));
                    reflectPhong = getUnitVect(reflectPhong);
                    double val = (dotMult(multVectScalar(ray.dir,-1),reflectPhong));///R.V
                    double phong = pow(max(val,0.0),shine);///R.V^k
                    lamb = max(lamb,0.0);
                    //phong = phong;

                    color[0]+= tempColor[0]*lights[i].color[0]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[1]+= tempColor[1]*lights[i].color[1]*(lamb*coEfficients[1]+phong*coEfficients[2]);
                    color[2]+= tempColor[2]*lights[i].color[2]*(lamb*coEfficients[1]+phong*coEfficients[2]);

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            if(level >= recursionDepth) return t;

            {
                point reflectRecur = sumVect(ray.dir,multVectScalar(normal,-2*dotMult(ray.dir,normal)));
                reflectRecur = getUnitVect(reflectRecur);
                point start = sumVect(intersectionPoint,multVectScalar(reflectRecur,1));
                Ray reflectionRay(start,reflectRecur);
                double *reflectedColor = new double[3];
                double tnew, tmin=99999;
                int nearest = -1;

                for(int i=0;i<objects.size();i++){
                    tnew = objects[i]->intersect(reflectionRay,reflectedColor,0);
                    if(tnew>0){
                        if(tnew<tmin){
                            nearest = i;
                            tmin = tnew;
                        }
                    }
                }

                if(nearest != -1){
                    objects[nearest]->intersect(reflectionRay,reflectedColor,level+1);
                    //if(level==recursionDepth){
                        color[0]+=reflectedColor[0]*coEfficients[3];
                        color[1]+=reflectedColor[1]*coEfficients[3];
                        color[2]+=reflectedColor[2]*coEfficients[3];
                    //}

                    for(int k=0; k<3;k++){
                        if(color[k]<0) color[k] = 0;
                        if(color[k]>1) color[k] = 1;
                    }
                }
            }

            for(int k=0; k<3;k++){
                if(color[k]<0) color[k] = 0;
                if(color[k]>1) color[k] = 1;
            }

            return t;
        }
    }
};


int screen_width = 600, screen_height = 600;
double viewAngle = 80;


void capture(){
    cout<<"capturing"<<endl;
    bitmap_image image(dimensions,dimensions);
    for(int i=0; i<dimensions;i++){
        for(int j=0; j<dimensions; j++){
            image.set_pixel(j,i,0,0,0);
        }
    }
    rightp = getUnitVect(rightp);
    look = getUnitVect(look);
    up = getUnitVect(up);

    struct point topleft,sum, curpixel;
    double planeDist = (screen_height/2.0)/tan((viewAngle/2.0)*pi/180);
    sum = sumVect(multVectScalar(rightp,-screen_width/2.0), multVectScalar(up,screen_height/2.0));
    sum = sumVect(sum, multVectScalar(look,planeDist));
    topleft = sumVect(sum,pos);

    double du = screen_width*1.0/dimensions;
    double dv = screen_height*1.0/dimensions;

    topleft = sumVect(topleft, multVectScalar(rightp,0.5*du));
    topleft = sumVect(topleft, multVectScalar(up,.05*dv));

    int nearest = -1;
    for(int i=0; i<dimensions; i++){
        for(int j=0; j<dimensions; j++){
            //cout<<"bhetore"<<endl;
            double t, tmin = 999999;
            curpixel = sumVect(topleft,multVectScalar(rightp,i*du));
            curpixel = sumVect(curpixel, multVectScalar(up,-j*dv));

            struct point raystart, raydir;
            raystart = pos;
            raydir = sumVect(curpixel, multVectScalar(pos,-1));
            raydir = getUnitVect(raydir);
            Ray ray(raystart, raydir);
            double *color = new double[3];

            for(int k=0; k<objects.size(); k++){
                t = objects[k]->intersect(ray, color, 0);

                //cout<<t<<endl;
                if(t>0){
                    if(tmin>t){
                        tmin = t;
                        nearest = k;
                        //cout<<"k: "<<k<<endl;
                    }
                }
            }

            if(nearest != -1){
                objects[nearest]->intersect(ray,color,1);
                image.set_pixel(i,j,color[0]*255,color[1]*255,color[2]*255);
            }
        }
    }
    cout<<"processing completed"<<endl;

    ///save
    image.save_image("G:\\Academics\\CSE 410\\Offline 3\\glut hw\\test.bmp");
    cout<<"capturing completed"<<endl;
}

void drawAxes(){
	if(drawaxes==1)
	{
		glColor3f(1.0, 1.0, 1.0);
		glBegin(GL_LINES);{
			glVertex3f( 100,0,0);
			glVertex3f(-100,0,0);

			glVertex3f(0,-100,0);
			glVertex3f(0, 100,0);

			glVertex3f(0,0, 100);
			glVertex3f(0,0,-100);
		}glEnd();
	}
}


void drawGrid(){
	int i;
	if(drawgrid==1)
	{
		glColor3f(0.6, 0.6, 0.6);
		glBegin(GL_LINES);{
			for(i=-8;i<=8;i++){

				if(i==0)
					continue;	//SKIP the MAIN axes

				//lines parallel to Y-axis
				glVertex3f(i*10, -90, 0);
				glVertex3f(i*10,  90, 0);

				//lines parallel to X-axis
				glVertex3f(-90, i*10, 0);
				glVertex3f( 90, i*10, 0);
			}
		}glEnd();
	}
}

void keyboardListener(unsigned char key, int x,int y){
	switch(key){

		case '1':
            rightp = rotateAroundAxis(rightp,up,3);
            look = rotateAroundAxis(look,up,3);
			break;
		case '2':
            rightp = rotateAroundAxis(rightp,up,-3);
            look = rotateAroundAxis(look,up,-3);
			break;
        case '3':
            look = rotateAroundAxis(look,rightp,3);
            up = rotateAroundAxis(up,rightp,3);
            break;
        case '4':
            look = rotateAroundAxis(look,rightp,-3);
            up = rotateAroundAxis(up,rightp,-3);
            break;
        case '5':
            rightp = rotateAroundAxis(rightp,look,3);
            up = rotateAroundAxis(up,look,3);
            break;
        case '6':
            rightp = rotateAroundAxis(rightp,look,-3);
            up = rotateAroundAxis(up,look,-3);
            break;
        case '0':
            capture();
            break;
        case 'q':
            if(rotate1<=45){
                rotate1+=3;
            }
            break;
        case 'w':
            if(rotate1>=-45){
                rotate1-=3;
            }
            break;
        case 'e':
            if(rotate2<=45){
                rotate2+=3;
            }
            break;
        case 'r':
            if(rotate2>=-45){
                rotate2-=3;
            }
            break;
        case 'a':
            if(rotate3<=45){
                rotate3+=3;
            }
            break;
        case 's':
            if(rotate3>=-45){
                rotate3-=3;
            }
            break;
        case 'd':
            if(rotate4<=45){
                rotate4+=3;
            }
            break;
        case 'f':
            if(rotate4>=-45){
                rotate4-=3;
            }
            break;

		default:
			break;
	}
}


void specialKeyListener(int key, int x,int y){
	switch(key){
		case GLUT_KEY_DOWN:		//down arrow key
			cameraHeight -= 3.0;
			pos = sumVect(pos,multVectScalar(look,-2));
			break;
		case GLUT_KEY_UP:		// up arrow key
			cameraHeight += 3.0;
			pos = sumVect(pos,multVectScalar(look,2));
			break;

		case GLUT_KEY_RIGHT:
			cameraAngle += 0.03;
            pos = sumVect(pos,multVectScalar(rightp,2));
			break;
		case GLUT_KEY_LEFT:
			cameraAngle -= 0.03;
			pos = sumVect(pos,multVectScalar(rightp,-2));
			break;

		case GLUT_KEY_PAGE_UP:
			pos = sumVect(pos,multVectScalar(up,2));
			break;
		case GLUT_KEY_PAGE_DOWN:
			pos = sumVect(pos,multVectScalar(up,-2));
			break;

		case GLUT_KEY_INSERT:
			break;

		case GLUT_KEY_HOME:
			break;
		case GLUT_KEY_END:
			break;

		default:
			break;
	}
}


void mouseListener(int button, int state, int x, int y){	//x, y is the x-y of the screen (2D)
	switch(button){
		case GLUT_LEFT_BUTTON:
			break;

		case GLUT_RIGHT_BUTTON:
			//........
			break;

		case GLUT_MIDDLE_BUTTON:
			//........
			break;

		default:
			break;
	}
}



void display(){

	//clear the display
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0,0,0,0);	//color black
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/********************
	/ set-up camera here
	********************/
	//load the correct matrix -- MODEL-VIEW matrix
	glMatrixMode(GL_MODELVIEW);

	//initialize the matrix
	glLoadIdentity();

	//now give three info
	//1. where is the camera (viewer)?
	//2. where is the camera looking?
	//3. Which direction is the camera's UP direction?

	//gluLookAt(100,100,100,	0,0,0,	0,0,1);
	//gluLookAt(200*cos(cameraAngle), 200*sin(cameraAngle), cameraHeight,		0,0,0,		0,0,1);
	//gluLookAt(0,0,200,	0,0,0,	0,1,0);
	gluLookAt(pos.x,pos.y,pos.z,	pos.x+look.x,pos.y+look.y,pos.z+look.z,	up.x,up.y,up.z);
	//gluLookAt(pos.x, pos.y, pos.z, pos.x + look.x, pos.y + look.y, pos.z + look.z, up.x, up.y, up.z);


	//again select MODEL-VIEW
	glMatrixMode(GL_MODELVIEW);


	/****************************
	/ Add your objects from here
	****************************/
	//add objects
    for(int i=0; i<objects.size(); i++){
        objects[i]->draw();
    }

    for(int i=0; i<lights.size(); i++){
        lights[i].draw();
    }
	//drawAxes();
	//drawGrid();

    //glColor3f(1,0,0);
    //drawSquare(10);

    //drawSS();

    //drawCircle(30,24);

    //drawCone(20,50,24);

	//drawSphere(30,24,20);




	//ADD this line in the end --- if you use double buffer (i.e. GL_DOUBLE)
	glutSwapBuffers();
}


void animate(){
	//codes for any changes in Models, Camera
	glutPostRedisplay();
}

void init(){
	//codes for initialization
	drawgrid=0;
	drawaxes=1;
	cameraHeight=150.0;
	cameraAngle=1.0;
	angle=0;
	/*********/
    up.x = 0;
    up.y = 0;
    up.z = 1;

    rightp.x = -1/sqrt(2);
    rightp.y = 1/sqrt(2);
    rightp.z = 0;

    look.x = -1/sqrt(2);
    look.y = -1/sqrt(2);
    look.z = 0;

    pos.x = 100;
    pos.y = 100;
    pos.z = 0;

	//clear the screen
	glClearColor(0,0,0,0);

	/************************
	/ set-up projection here
	************************/
	//load the PROJECTION matrix
	glMatrixMode(GL_PROJECTION);

	//initialize the matrix
	glLoadIdentity();

	//give PERSPECTIVE parameters
	gluPerspective(80,	1,	1,	1000.0);
	//field of view in the Y (vertically)
	//aspect ratio that determines the field of view in the X direction (horizontally)
	//near distance
	//far distance
}

void loadData(){
    string filename = "G:\\Academics\\CSE 410\\Offline 3\\glut hw\\scene_test.txt";
    fstream file;
    string line;
    int nObjects, nLights;

    file.open(filename.c_str(), ios::in|ios::out|ios::app);

    file>>recursionDepth;
    file>>dimensions;
    file>>nObjects;

    Object *temp;
    temp = new Floor(1000,20);
    objects.push_back(temp);

    cout<<recursionDepth<<endl;
    cout<<dimensions<<endl;
    cout<<nObjects<<endl;

    while(nObjects--){
        string line;
        file >> line;
        if( line == "sphere"){
            double x,y,z, rad, r,g,b,amb,diff,spec,recref;
            int shine;
            file>>x>>y>>z>>rad>>r>>g>>b>>amb>>diff>>spec>>recref>>shine;

            temp = new Sphere(x,y,z, rad);
            temp->setColor(r,g,b);
            temp->setShine(shine);
            temp->setCoEfficients(amb,diff,spec,recref);

            objects.push_back(temp);
        }else if(line == "triangle"){
            struct point vertexA, vertexB, vertexC;
            double x1,y1,z1,x2,y2,z2,x3,y3,z3,r,g,b,amb,diff,spec,recref;
            int shine;
            file>>x1>>y1>>z1>>x2>>y2>>z2>>x3>>y3>>z3>>r>>g>>b>>amb>>diff>>spec>>recref>>shine;

            vertexA.x = x1;
            vertexA.y = y1;
            vertexA.z = z1;

            vertexB.x = x2;
            vertexB.y = y2;
            vertexB.z = z2;

            vertexC.x = x3;
            vertexC.y = y3;
            vertexC.z = z3;

            temp = new Triangle(vertexA,vertexB,vertexC);
            temp->setColor(r,g,b);
            temp->setCoEfficients(amb,diff,spec,recref);
            temp->setShine(shine);
            objects.push_back(temp);

        }else if(line == "general"){
            double a,b,c,d,e,f,g,h,i,j,x,y,z,length,width, height,color_r,color_g,color_b,amb,diff,spec,recref;
            int shine;
            struct point refpoint;
            file>>a>>b>>c>>d>>e>>f>>g>>h>>i>>j>>x>>y>>z>>length>>width>>height>>color_r>>color_g>>color_b>>amb>>diff>>spec>>recref>>shine;
            refpoint.x = x;
            refpoint.y = y;
            refpoint.z = z;
            temp = new General(a,b,c,d,e,f,g,h,i,j,length,width,height);
            temp->setColor(color_r,color_g,color_b);
            temp->setCoEfficients(amb,diff,spec,recref);
            temp->ref_point = refpoint;
            temp->setShine(shine);

            objects.push_back(temp);
        }
    }

    file>>nLights;

    while(nLights--){
        double x,y,z,r,g,b;
        file>>x>>y>>z>>r>>g>>b;

        struct point pos;
        pos.x = x;
        pos.y = y;
        pos.z = z;
        Light temp = Light(pos, r,g,b);
        lights.push_back(temp);
    }

    /*for(int i=0; i<objects.size(); i++){
        objects[i]->print();
    }*/
}

int main(int argc, char **argv){

    loadData();

	glutInit(&argc,argv);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);	//Depth, Double buffer, RGB color

	glutCreateWindow("My OpenGL Program");

	init();

	glEnable(GL_DEPTH_TEST);	//enable Depth Testing

	glutDisplayFunc(display);	//display callback function
	glutIdleFunc(animate);		//what you want to do in the idle time (when no drawing is occuring)

	glutKeyboardFunc(keyboardListener);
	glutSpecialFunc(specialKeyListener);
	glutMouseFunc(mouseListener);

	glutMainLoop();		//The main loop of OpenGL

	///memory management
	for(int i=0; i<objects.size(); i++){
        delete objects[i];
	}
	objects.clear();
	lights.clear();

	return 0;
}
