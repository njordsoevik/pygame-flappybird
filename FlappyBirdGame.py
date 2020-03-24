#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:05:34 2019

@author: nsoevik
"""
import pygame
import neat
import time
import random
import os
pygame.font.init()
cwd = os.getcwd()
data_dir=cwd+'config_forwardfeed.txt'
win_w=600
win_h=750


'''
inputs = Bird y, top pipe, bottom pipe
output = jump
activation function = tanh
population size (NEAT Specific) = 100
fitness function = how far it goes
max gen = 30
'''

bird_imgs= [
 pygame.transform.scale2x(pygame.image.load(os.path.join(data_dir,'bird1.png')))
,pygame.transform.scale2x(pygame.image.load(os.path.join(data_dir,'bird2.png')))
,pygame.transform.scale2x(pygame.image.load(os.path.join(data_dir,'bird3.png')))]

pipe_img=pygame.transform.scale2x(pygame.image.load(os.path.join(data_dir,'pipe.png')))
base_img=pygame.transform.scale2x(pygame.image.load(os.path.join(data_dir,'base.png')))
bg_img=pygame.transform.scale2x(pygame.image.load(os.path.join(data_dir,'bg.png')))
STAT_FONT = pygame.font.SysFont("comicsans",50)


class Bird:
    IMGS = bird_imgs
    MAX_ROTATION=25
    ROT_VEL=20
    ANIMATION_TIME=5
    
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tilt=0
        self.tick_count=0
        self.vel=0
        self.height=self.y
        self.img_count=0
        self.img=self.IMGS[0]
        
    def jump(self):
        self.vel=-10.5
        self.tick_count=0
        self.height=self.y
    '''when jumping, set upwards velocity and tick count to 0, set height to'''
    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    def draw(self, win):
        self.img_count += 1
        
        if self.img_count<self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count<self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count<self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count<self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count<self.ANIMATION_TIME*4+1:
            self.img = self.IMGS[0]
            self.img_count=0
            
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
            
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x,self.y)).center)
        win.blit(rotated_image, new_rect.topleft)
    
    def get_mask(self): #list of where pixels are in a box, for collision
        return pygame.mask.from_surface(self.img)
    
class Pipe:
    GAP=200
    VEL=5
    
    def __init__(self,x):
        self.x=x
        self.height=0
        
        self.top=0
        self.bottom=0
        self.PIPE_TOP= pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM=pipe_img
        
        self.passed = False
        self.set_height()
        
    def set_height(self):
        self.height=random.randrange(40,450)
        self.top=self.height - self.PIPE_TOP.get_height()
        self.bottom=self.height + self.GAP
        
    def move(self):
        self.x-=self.VEL
        
    def draw(self,win):
        win.blit(self.PIPE_TOP,(self.x,self.top))
        win.blit(self.PIPE_BOTTOM, (self.x,self.bottom))
        
    def collide(self, bird):
        bird_mask=bird.get_mask()
        top_mask=pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask=pygame.mask.from_surface(self.PIPE_BOTTOM)
        
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        
        b_point = bird_mask.overlap(bottom_mask,bottom_offset) # no collide return none
        t_point = bird_mask.overlap(top_mask,top_offset)# no collide return none
        
        if b_point or t_point:
            return True
        
        return False

class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    img=base_img
    
    def __init__(self,y):
        self.y=y
        self.x1=0
        self.x2=self.WIDTH
        
    def move(self):
        self.x1-=self.VEL
        self.x2-=self.VEL
        
        if self.x1+self.WIDTH<0:
            self.x1=self.x2+self.WIDTH
        if self.x2+self.WIDTH<0:
            self.x2=self.x1+self.WIDTH   
        
    def draw(self,win):
        win.blit(self.img,(self.x1, self.y))
        win.blit(self.img,(self.x2, self.y))
    
def draw_window(win,birds,pipes,base,score):
    win.blit(bg_img, (0,0))
    
    for pipe in pipes:
        pipe.draw(win)
        
    text = STAT_FONT.render("Score: "+ str(score), 1,(255,255,255))
    win.blit(text,(win_w - 10 - text.get_width(),10))
    
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()
    
    
def main(genomes,config):
    nets=[]
    ge=[]
    birds=[]
    
    for _,g in genomes: # genomes is a tuple of id and object, we just want object
        net = neat.nn.FeedForwardNetwork.create(g,config) # create the network
        nets.append(net) # append this network to network list
        birds.append(Bird(230,350)) # add a bird to our list for network
        g.fitness=0 # set initiall fitness for genome
        ge.append(g) # add genome to ge list
    
    
    base=Base(730)
    pipes=[Pipe(700)]
    score=0
    
    win=pygame.display.set_mode((win_w,win_h))
    clock=pygame.time.Clock()
    run=True
    while run:
        
        clock.tick(30)
        for event in pygame.event.get():#get action, so we can quit when hitting X
            if event.type == pygame.QUIT:
                run = False       
                pygame.quit()
                quit()
        
        pipe_ind= 0 
        if len(birds)>0:
            if len(pipes)> 1 and birds[0].x> pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind=1
        else:
            run=False
            break
        
        for x,bird in enumerate(birds):
            bird.move()
            ge[x].fitness+=0.1
            
            output = nets[x].activate((bird.y,abs(bird.y - pipes[pipe_ind].height),abs(bird.y - pipes[pipe_ind].bottom))) # bird height, distance between top pipe and bird, y position of bottom pipe 
            
            if output[0]>0.5:
                bird.jump()

        base.move()
        remove_list=[]
        add_pipe=False
        for pipe in pipes:
            for x,bird in enumerate(birds):     
                if pipe.collide(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x) # remove everything associated with the bird

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed=True
                    add_pipe=True
                        
            if pipe.x+pipe.PIPE_TOP.get_width() <0:
                remove_list.append(pipe)
                
            pipe.move()
            
        if add_pipe:
            score+=1
            for g in ge:
                g.fitness += 5 # add fitness to remaining birds
            pipes.append(Pipe(600))
        
        for r in remove_list:
            pipes.remove(r)
        
        for x,bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y<0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
        
        draw_window(win,birds,pipes,base,score)
    

def run(config_path):
    config=neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path) # properties from settings file
    
    p=neat.Population(config) # create population based on config
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #optional to give us some output on statistics

    winner = p.run(main,25) # set the fitness function, and popualtion
    
if __name__=="__main__":
    config_path=os.path.join(data_dir,'config_forwardfeed.txt')
    run(config_path)
