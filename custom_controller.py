import sys, math, json
import os

from matplotlib.pyplot import locator_params 

from Baseline_snakeoil.client import Track, TrackSection
from Baseline_snakeoil import snakeoil
import utils

dir_path = os.path.dirname(os.path.realpath(__file__))

class CustomController:
    
    def __init__(self, port=None, parameters = None, parameters_from_file = True, parameter_file = "\Baseline_snakeoil\default_parameters"):
        self.port = port
        self.parameters = parameters
        self.parameters_from_file = parameters_from_file
        self.parameter_file = parameter_file        

        self.target_speed= 0 
        self.lap= 0 
        self.prev_distance_from_start= 1 
        self.learn_final= False 
        self.opHistory= list() 
        self.trackHistory= [0] 
        self.TRACKHISTORYMAX= 50 
        self.secType= 0 
        self.secBegin= 0 
        self.secMagnitude= 0 
        self.secWidth= 0 
        self.sangs= [-45,-19,-12,-7,-4,-2.5,-1.7,-1,-.5,0,.5,1,1.7,2.5,4,7,12,19,45]
        self.sangsrad= [(math.pi*X/180.0) for X in self.sangs]
        self.badness= 0
    
    # (P,S['rpm'],S['gear'],R['clutch'],S['rpm'],S['speedX'],self.target_speed,tick)
    def automatic_transimission(self, P,r,g,c,rpm,sx,ts,tick):
        clutch_releaseF= .05 
        ng,nc= g,c 
        if ts < 0 and g > -1: 
            ng= -1
            nc= 1
        elif ts>0 and g<0:
            ng= g+1
            nc= 1
        elif c > 0:
            if g: 
                nc= c - clutch_releaseF 
            else: 
                if ts < 0:
                    ng= -1 
                else:
                    ng= 1 
        elif not tick % 50 and sx > 20:
            pass 
        elif g==6 and rpm<P['dnsh5rpm']: 
            ng= g-1 
            nc= 1
        elif g==5 and rpm<P['dnsh4rpm']: 
            ng= g-1 
            nc= 1
        elif g==4 and rpm<P['dnsh3rpm']:
            ng= g-1 
            nc= 1
        elif g==3 and rpm<P['dnsh2rpm']:
            ng= g-1 
            nc= 1
        elif g==2 and rpm<P['dnsh1rpm']:
            ng= g-1 
            nc= 1
        elif g==5 and rpm>P['upsh6rpm']: 
            ng= g+1
            nc= 1
        elif g==4 and rpm>P['upsh5rpm']: 
            ng= g+1
            nc= 1
        elif g==3 and rpm>P['upsh4rpm']: 
            ng= g+1
            nc= 1
        elif g==2 and rpm>P['upsh3rpm']: 
            ng= g+1
            nc= 1
        elif g==1 and rpm>P['upsh2rpm']: 
            ng= g+1
            nc= 1
        elif not g:
            ng= 1
            nc= 1
        else:
            pass
        return ng,nc

    def find_slip(self, wsv_list):
        w1,w2,w3,w4= wsv_list
        if w1:
            slip= (w3+w4) - (w1+w2)
        else:
            slip= 0
        return slip
    
    # S['track'], S['angle']
    def track_sensor_analysis(self, t,a):
        alpha= 0
        sense= 1 
        farthest= None,None 
        ps= list()
        realt= list()
        sangsradang= [(math.pi*X/180.0)+a for X in self.sangs] 
        for n,sang in enumerate(sangsradang):
            x,y= t[n]*math.cos(sang),t[n]*-math.sin(sang)
            if float(x) > 190:
                alpha= math.pi
            else:
                ps.append((x,y))
                realt.append(t[n])
        firstYs= [ p[1] for p in ps[0:3] ]
        lastYs= [ p[1] for p in ps[-3:] ]
        straightnessf= abs(1- abs(min(firstYs))/max(.0001,abs(max(firstYs))))
        straightnessl= abs(1- abs(min(lastYs))/max(.0001,abs(max(lastYs))))
        straightness= max(straightnessl,straightnessf)
        farthest= realt.index(max(realt))
        ls= ps[0:farthest] 
        rs= ps[farthest+1:] 
        rs.reverse() 
        if farthest > 0 and farthest < len(ps)-1: 
            beforePdist= t[farthest-1]
            afterPdist=  t[farthest+1]
            if beforePdist > afterPdist: 
                sense= -1
                outsideset= ls
                insideset= rs
                ls.append(ps[farthest]) 
            else:                        
                outsideset= rs
                insideset= ls
                rs.append(ps[farthest]) 
        else: 
            if ps[0][0] > ps[-1][0]: 
                ps.reverse()
                farthest= (len(ps)-1) - farthest 
            if ps[0][1] > ps[-1][1]: 
                sense= -1
                outsideset= ls
                insideset= rs
            else: 
                outsideset= rs
                insideset= ls
        maxpdist= 0
        if not outsideset:
            return (0,a,2)
        nearx,neary= outsideset[0][0],outsideset[0][1]
        farx,fary= outsideset[-1][0],outsideset[-1][1]
        cdeltax,cdeltay= (farx-nearx),(fary-neary)
        c= math.sqrt(cdeltax*cdeltax + cdeltay*cdeltay)
        for p in outsideset[1:-1]: 
            dx1= p[0] - nearx
            dy1= p[1] - neary
            dx2= p[0] - farx
            dy2= p[1] - fary
            a= math.sqrt(dx1*dx1+dy1*dy1)
            b= math.sqrt(dx2*dx2+dy2*dy2)
            pdistances= a + b
            if pdistances > maxpdist:
                maxpdist= pdistances
                inflectionp= p  
                ia= a 
                ib= b 
        if maxpdist and not alpha:
            infleX= inflectionp[0]
            preprealpha= 2*ia*ib
            if not preprealpha: preprealpha= .00000001 
            prealpha= (ia*ia+ib*ib-c*c)/preprealpha
            if prealpha > 1: alpha= 0
            elif prealpha < -1: alpha= math.pi
            else:
                alpha= math.acos(prealpha)
            turnsangle= sense*(180-(alpha *180 / math.pi))
        else:
            infleX= max(t)
            turnsangle= self.sangs[t.index(infleX)]
        return (infleX,turnsangle,straightness)

    #P,S['distFromStart'],S['track'],S['trackPos'],S['speedX'],S['speedY'],R['steer'],S['angle'],infleX,infleA)
    def speed_planning(self, P,d,t,tp,sx,sy,st,a,infleX,infleA):
        # take the max distance from the track edge
        cansee= max(t[2:17])
        if cansee > 0:
            carmax= P['carmaxvisib'] * cansee 
        else:
            carmax= 69
        if cansee <0: 
            return P['backontracksx'] 
        if cansee > 190 and abs(a)<.1:
            return carmax 
        if t[9] < 40: 
            return P['obviousbase'] + t[9] * P['obvious']
       
        if infleA:
            willneedtobegoing= 600-180.0*math.log(abs(infleA))
            willneedtobegoing= max(willneedtobegoing,P['carmin']) 
        else: 
            willneedtobegoing= carmax
       
        brakingpacecut= 150 
        if sx > brakingpacecut:
            brakingpace= P['brakingpacefast']
        else:
            brakingpace= P['brakingpaceslow']
            
        base= min(infleX * brakingpace + willneedtobegoing,carmax)
        base= max(base,P['carmin']) 
        if st<P['consideredstr8']: 
            return base
        uncoolsy= abs(sy)/sx
        syadjust= 2 - 1 / P['oksyp'] * uncoolsy
        return base * syadjust 

    def damage_speed_adjustment(self, d):
        dsa= 1
        if d > 1000: dsa=.98
        if d > 2000: dsa=.96
        if d > 3000: dsa=.94
        if d > 4000: dsa=.92
        if d > 5000: dsa=.90
        if d > 6000: dsa=.88
        return dsa

    def jump_speed_adjustment(self, z):
        offtheground= snakeoil.clip(z-.350,0,1000)
        jsa= offtheground * -800
        return jsa

    def traffic_speed_adjustment(self, os,sx,ts,tsen):
        if not self.opHistory: 
            self.opHistory= os 
            return 0 
        tsa= 0 
        mpn= 0 
        sn=  min(os[17],os[18])  
        if sn > tsen[9] and tsen[9]>0: 
            return 0                   
        if sn < 15:
            sn=  min(sn , os[16],os[19])  
        if sn < 8:
            sn=  min(sn , os[15],os[20])  
        sn-= 5 
        if sn<3: 
            self.opHistory= os 
            return -ts 
        opn= mpn+sn 
        mpp= mpn - sx/180 
        sp= min(self.opHistory[17],self.opHistory[18]) 
        if sp < 15:
            sp=  min(sp , os[16],os[19])  
        if sp < 8:
            sp=  min(sn , os[15],os[20])  
        sp-= 5 
        self.opHistory= os 
        opp= mpp+sp 
        osx= (opn-opp) * 180 
        osx= snakeoil.clip(osx,0,300) 
        if osx-sx > 0: return 0 
        max_tsa= osx - ts 
        max_worry= 80 
        full_serious= 20 
        if sn > max_worry:
            seriousness= 0
        elif sn < full_serious:
            seriousness= 1
        else:
            seriousness= (max_worry-sn)/(max_worry-full_serious)
        tsa= max_tsa * seriousness
        tsa= snakeoil.clip(tsa,-ts,0) 
        return tsa

    # P,R['steer'],S['trackPos'],S['angle']
    def steer_centeralign(self, P,sti,tp,a,ttp=0):
        
        pointing_ahead= abs(a) < P['pointingahead'] 
        # tp represents if the car is inside or outside the track
        # value normalized with respect to the track width. <-1 or >1 the car is outside the track
        onthetrack= abs(tp) < P['sortofontrack']
        offrd= 1
        if not onthetrack:
            offrd= P['offroad']
        if pointing_ahead:
            sto= a 
        else:
            sto= a * P['backward']
        ttp*= 1-a 
        sto+= (ttp - min(tp,P['steer2edge'])) * P['s2cen'] * offrd 
        return sto 

    def speed_appropriate_steer(self, P,sto,sx):
        # sx -> speed x
        if sx > 0:
            stmax=  max(P['sxappropriatest1']/math.sqrt(sx)-P['sxappropriatest2'],P['safeatanyspeed'])
        else:
            stmax= 1
        return snakeoil.clip(sto,-stmax,stmax)

    #(P,R['steer'],S['trackPos'],S['angle'],S['track'],S['speedX'],infleX,infleA,straightness)
    def steer_reactive(self, P,sti,tp,a,t,sx,infleX,infleA,str8ness):
        if abs(a) > .6: 
            return self.steer_centeralign(P,sti,tp,a)
        # take the max distance from the edge of the track
        maxsen= max(t)
        ttp= 0
        aadj= a
        # If you are on track
        if maxsen > 0 and abs(tp) < .99:
            MaxSensorPos= t.index(maxsen)
            # take the angle of the sensor that returns the max distance [rad]
            MaxSensorAng= self.sangsrad[MaxSensorPos]
            #sensangF= -.9  
            sensangF = P['sensang']
            aadj= MaxSensorAng * sensangF
            if maxsen < 40:
                ttp= MaxSensorAng * - P['s2sen'] / maxsen
            else: 
                if str8ness < P['str8thresh'] and abs(infleA)>P['ignoreinfleA']:
                    ttp= -abs(infleA)/infleA
                    aadj= 0 
                else:
                    ttp= 0
            senslimF= .031 
            ttp= snakeoil.clip(ttp,tp-senslimF,tp+senslimF)
        else: 
            aadj= a
            if tp:
                ttp= .94 * abs(tp) / tp
            else:
                ttp= 0
        sto= self.steer_centeralign(P,sti,tp,aadj,ttp)
        return self.speed_appropriate_steer(P,sto,sx)

    def traffic_navigation(self, os, sti):
        sto= sti 
        c= min(os[4:32]) 
        cs= os.index(c)  
        if not c: c= .0001
        if min(os[18:26])<7:
            sto+= .5/c
        if min(os[8:17])<7:
            sto-= .5/c
        if cs == 17:
            sto+= .1/c
        if cs == 18:
            sto-= .1/c
        if .1 < os[17] < 40:
            sto+= .01
        if .1 < os[18] < 40:
            sto-= .01
        return sto

    # P,R['clutch'],slip,S['speedX'],S['speedY'],S['gear']
    def clutch_control(self, P,cli,sl,sx,sy,g):
        if abs(sx) < .1 and not cli: 
            return 1  
        clo= cli-.2 
        clo+= sl/P['clutchslip']
        clo+= sy/P['clutchspin']
        return clo

    # P,R['accel'],self.target_speed,S['speedX'],slip, S['speedY'],S['angle'],R['steer']
    def throttle_control(self, P,ai,ts,sx,sl,sy,ang,steer):
        ao= ai 
        if ts < 0:
            tooslow= sx-ts 
        else:
            okmaxspeed4steer= P['stst']*steer*steer-P['st']*steer+P['stC']
            if steer> P['fullstis']:
                ts=P['fullstmaxsx']
            else:
                ts= min(ts,okmaxspeed4steer)
            tooslow= ts-sx 
        ao= 2 / (1+math.exp(-tooslow)) -1
        
        # reduce ao if the car is slipping
        ao-= abs(sl) * P['slipdec'] 
        spincut= P['spincutint']-P['spincutslp']*abs(sy)
        spincut= snakeoil.clip(spincut,P['spincutclip'],1)
        ao*= spincut
        ww= abs(ang)/P['wwlim']
        wwcut=  min(ww,.1)
        
        if ts>0 and sx >5:
            ao-= wwcut
        if ao > .8: ao= 1

        return ao
    
    #(P,R['brake'],S['speedX'],S['speedY'],self.target_speed,skid
    def brake_control(self, P,bi,sx,sy,ts,sk):
        bo= bi 
        toofast= sx-ts
        # the car speed is lower than the target speed
        # the brake is set to zero
        if toofast < 0: 
            return 0
        # the car speed is higher than the target speed
        if toofast: 
            # direttamente proporzionale alla differenza di velocità
            # inversamente proporzionale allo slittamentos
            bo+= P['brake'] * toofast / max(1,abs(sk))
            #bo=1
        if sk > P['seriousABS']: bo=0 
        if sx < 0: bo= 0 
        if sx < -.1 and ts > 0:  
            bo+= .05
        sycon= 1
        if sy:
            sycon= min(1,  P['sycon2']-P['sycon1']*math.log(abs(sy))  )
        
        # output [0,1]
        return min(bo,sycon)

    def iberian_skid(self, wsv,sx):
        speedps= sx/3.6
        sxshouldbe= sum( [ [.3179,.3179,.3276,.3276][x] * wsv[x] for x in range(3) ] ) / 4.0
        return speedps-sxshouldbe
    
    # P,S['wheelSpinVel'],S['speedX']
    def skid_severity(self, P,wsv_list,sx):
        skid= 0
        avgws= sum(wsv_list)/4 
        if avgws:
            skid= P['skidsev1']*sx/avgws - P['wheeldia'] 
        return skid

    def car_might_be_stuck(self, sx,a,p):
        if p > 1.2 and a < -.5:
            return True
        if p < -1.2 and a > .5:
            return True
        if sx < 3: 
            return True
        return False 

    def car_is_stuck(self, sx,t,a,p,fwdtsen,ts):
        if fwdtsen > 5 and ts > 0: 
            return False
        if abs(a)<.5 and abs(p)<2 and ts > 0: 
            return False
        if t < 100: 
            return False
        return True

    def learn_track(self, st,a,t,dfs):
        NOSTEER= 0.02
        self.T.laplength= max(dfs,self.T.laplength)
        if len(self.trackHistory) >= self.TRACKHISTORYMAX:
            self.trackHistory.pop(0) 
        self.trackHistory.append(st)
        steer_sma= sum(self.trackHistory)/len(self.trackHistory) 
        if abs(steer_sma) > NOSTEER: 
            secType_now= abs(steer_sma)/steer_sma
            if self.secType != secType_now: 
                self.T.sectionList.append( TrackSection(self.secBegin,dfs, self.secMagnitude, self.secWidth,0) )
                self.secMagnitude= 0 
                self.secWidth= 0 
                self.secType= secType_now 
                self.secBegin= dfs 
            self.secMagnitude+= st 
        else: 
            if self.secType: 
                self.T.sectionList.append( TrackSection(self.secBegin,dfs, self.secMagnitude, self.secWidth,0) )
                self.secMagnitude= 0 
                self.secWidth= 0 
                self.secType= 0 
                self.secBegin= dfs 
        if not self.secWidth and abs(a) < NOSTEER:
            self.secWidth= t[0]+t[-1] 

    def learn_track_final(self, dfs):
        #global secType
        self.T.sectionList.append( TrackSection(self.secBegin,dfs, self.secMagnitude, self.secWidth, self.badness) )

    def drive(self, c, tick):
        S,R,P= c.S.d,c.R.d,c.P

        self.badness= S['damage']-self.badness
        # compute the skid severity
        skid= self.skid_severity(P,S['wheelSpinVel'],S['speedX'])
        if skid>1:
            self.badness+= 15
        # check if the car may be stuck
        if self.car_might_be_stuck(S['speedX'],S['angle'],S['trackPos']):
            S['stucktimer']= (S['stucktimer']%400) + 1
            # check if the car is stuck
            # S['track'][9], it's the ninth sensor value, distance from the car and the track edge
            if self.car_is_stuck(S['speedX'],S['stucktimer'],S['angle'],
                            S['trackPos'],S['track'][9],self.target_speed):
                self.badness+= 100
                R['brake']= 0 
                if self.target_speed > 0:
                    self.target_speed= -40
                else:
                    self.target_speed= 40
        else: 
            S['stucktimer']= 0
        if S['z']>4: 
            self.badness+= 20
        infleX,infleA,straightness= self.track_sensor_analysis(S['track'],S['angle'])
        if self.target_speed>0:
            if c.stage:
                if not S['stucktimer']:
                    # return target speed, [km/h]
                    self.target_speed= self.speed_planning(P,S['distFromStart'],S['track'],S['trackPos'],
                                            S['speedX'],S['speedY'],R['steer'],S['angle'],
                                            infleX,infleA)
                self.target_speed+= self.jump_speed_adjustment(S['z'])
                if c.stage > 1: 
                    self.target_speed+= self.traffic_speed_adjustment(
                            S['opponents'],S['speedX'],self.target_speed,S['track'])
                self.target_speed*= self.damage_speed_adjustment(S['damage'])
            else:
                if self.lap > 1 and self.T.usable_model:
                    self.target_speed= self.speed_planning(P,S['distFromStart'],S['track'],S['trackPos'],
                                            S['speedX'],S['speedY'],R['steer'],S['angle'],
                                            infleX,infleA)
                    self.target_speed*= self.damage_speed_adjustment(S['damage'])
                else:
                    print("Set target speed to 50")
                    self.target_speed= 50
        self.target_speed= min(self.target_speed,333)
        caution= 1
        # In unknown this is always skipped 
        if self.T.usable_model:
            snow= self.T.section_in_now(S['distFromStart'])
            snext= self.T.section_ahead(S['distFromStart'])
            if snow:
                if snow.self.badness>100: caution= .80
                if snow.self.badness>1000: caution= .65
                if snow.self.badness>10000: caution= .4
                if snext:
                    if snow.end - S['distFromStart'] < 200: 
                        if snext.self.badness>100: caution= .90
                        if snext.self.badness>1000: caution= .75
                        if snext.self.badness>10000: caution= .5
        self.target_speed*= caution
        # In unknown this if is true
        if self.T.usable_model or c.stage>1:
            # if the car is not on the axis
            if abs(S['trackPos']) > 1:
                # it brings the car towards the center
                s= self.steer_centeralign(P,R['steer'],S['trackPos'],S['angle'])
                self.badness+= 1
            else:
                s= self.steer_reactive(P,R['steer'],S['trackPos'],S['angle'],S['track'],
                                                    S['speedX'],infleX,infleA,straightness)
        else:
            # it brings the car towards the center
            s= self.steer_centeralign(P,R['steer'],S['trackPos'],S['angle'])
        # set the steer value to send to the server
        R['steer']= s
        
        if S['stucktimer'] and S['distRaced']>20:
            # if you are going in reverse.
            # invert the rotation, in order to align with the track
            if self.target_speed<0:
                R['steer']= -S['angle']
        # In unknown this if is true
        if c.stage > 1: 
            if self.target_speed < 0:
                # S['opponents'] distance from the closest opponent.
                self.target_speed*= snakeoil.clip(S['opponents'][0]/20,  .1, 1)
                self.target_speed*= snakeoil.clip(S['opponents'][35]/20, .1, 1)
            else:
                # override what has been decided before.
                # set the steer based on the current car speed: S['speedX']+50
                # the desired steer based on the presence of the opponents: self.traffic_navigation(S['opponents'], R['steer'])
                R['steer']= self.speed_appropriate_steer(P, 
                                                        self.traffic_navigation(S['opponents'], R['steer']),
                                                        S['speedX']+50)
        
        if not S['stucktimer']:
            self.target_speed= abs(self.target_speed) 
        
        # slipping
        slip= self.find_slip(S['wheelSpinVel'])

        R['accel']= self.throttle_control(P,R['accel'],self.target_speed,S['speedX'],slip,
                                    S['speedY'],S['angle'],R['steer'])
        
        if R['accel'] < .01:
            R['brake']= self.brake_control(P,R['brake'],S['speedX'],S['speedY'],self.target_speed,skid)
        else:
            R['brake']= 0
        
        R['gear'],R['clutch']= self.automatic_transimission(P,
            S['rpm'],S['gear'],R['clutch'],S['rpm'],S['speedX'],self.target_speed,tick)
        
        #R['clutch']= self.clutch_control(P,R['clutch'],slip,S['speedX'],S['speedY'],S['gear'])
        
        if S['distRaced'] < S['distFromStart']: 
            self.lap= 0
        
        if self.prev_distance_from_start > S['distFromStart'] and abs(S['angle'])<.1:
            self.lap+= 1
        
        self.prev_distance_from_start= S['distFromStart']
        if not self.lap: 
            self.T.laplength= max(S['distFromStart'],self.T.laplength)
        elif self.lap == 1 and not self.T.usable_model: 
            self.learn_track(R['steer'],S['angle'],S['track'],S['distFromStart'])
        elif c.stage == 3:
            pass 
        else: 
            if not self.learn_final: 
                self.learn_track_final(self.T.laplength)
                self.T.post_process_track()
                self.learn_final= True
            if self.T.laplength:
                properlap= S['distRaced']/self.T.laplength
            else:
                properlap= 0
            if c.stage == 0 and self.lap < 4: 
                self.T.record_badness(self.badness,S['distFromStart'])
        
        S['targetSpeed']= self.target_speed 
        self.target_speed= 70 
        self.badness= S['damage']
        return

    def initialize_car(self, c):
        R= c.R.d
        R['gear']= 1 
        R['steer']= 0 
        R['brake']= 1 
        R['clutch']= 1 
        R['accel']= .22 
        R['focus']= 0 
        c.respond_to_server() 

    def run_controller(self, plot_history = False):
        # load parameters
        if self.parameters_from_file:
            pfile= open(dir_path + self.parameter_file,'r') 
            P = json.load(pfile)
        else:
            P = self.parameters

        self.T = Track()
        self.C = snakeoil.Client(p=self.port, P=P)
            
        if self.C.stage == 1 or self.C.stage == 2:
            try:
                self.T.load_track(self.C.trackname)
            except:
                print(f"Could not load the track: {self.C.trackname}") 
                sys.exit()
            print("Track loaded!")

        self.initialize_car(self.C)
        self.C.S.d['stucktimer']= 0
        self.C.S.d['targetSpeed']= 0

        history_speed = {}    
        history_damage = {}
        history_lap_time = {}
        
        lap_cnt = 1
        last_lap_time_prev = 0.0

        for step in range(self.C.maxSteps,0,-1):
            return_code = self.C.get_servers_input()
            
            try:   
                if self.C.S.d['lastLapTime'] != last_lap_time_prev:
                    # store the lap time
                    history_lap_time[lap_cnt] = self.C.S.d['lastLapTime']    
                    last_lap_time_prev = self.C.S.d['lastLapTime']
                    lap_cnt +=1
                
                history_speed[lap_cnt]
                if lap_cnt >= 1:
                    # store the history
                    history_speed[lap_cnt].append(math.sqrt(self.C.S.d['speedX']**2+self.C.S.d['speedY']**2+self.C.S.d['speedZ']**2))
                    history_damage[lap_cnt].append(self.C.S.d['damage'])
            except KeyError as e:
                if lap_cnt >= 1:
                    # initialize the history for the current lap
                    history_speed[lap_cnt] = [math.sqrt(self.C.S.d['speedX']**2+self.C.S.d['speedY']**2+self.C.S.d['speedZ']**2)]
                    history_damage[lap_cnt] = [self.C.S.d['damage']]

            if return_code == snakeoil.RACE_ENDED:
                #print("Race ended")
                break
        
            self.drive(self.C, step)
            self.C.respond_to_server()
        # save the history
        #print(f"Lap time: {history_lap_time}")
        if plot_history:
            utils.plot_history(history_lap_time, history_speed, history_damage)

        if not self.C.stage:
            self.T.write_track(self.C.trackname) 
        self.C.R.d['meta']= 1
        self.C.respond_to_server()
        #C.shutdown()
        
        return history_lap_time, history_speed, history_damage

if __name__ == "__main__":
    controller = CustomController()
    controller.run_controller()