# -*- coding: utf-8 -*-
"""
Because this game never dies and deserved to meet Python and AI.

The original game by Espen Skoglund (http://hol.abime.net/3853) was born in the early 90s on the Commodore Amiga. That was the great time of MC68000 Assembly.

Around 2000 we made a PC version (https://github.com/devpack/mayhem) of the game in C++.

It was then ported to Raspberry Pi by Martin O'Hanlon (https://github.com/martinohanlon/mayhem-pi), even new gfx levels were added.

----

It was early time this game had its own Python version. Pygame (https://www.pygame.org/docs) SDL wrapper has been used as backend.

The ultimate goal porting it to Python is to create a friendly AI environment (like Gym (https://gym.openai.com/envs/#atari)) which could easily be used with Keras (https://keras.io) deep learning framework. AI players in Mayhem are coming !

Anthony Prieur
anthony.prieur@gmail.com
"""

import os, sys, argparse, random, math, time
from random import randint

import pygame
from pygame.locals import *
from pygame import gfxdraw

try:
    import cPickle as pickle
except ImportError:
    import pickle

# -------------------------------------------------------------------------------------------------

DEBUG_SCREEN = True # print debug info on the screen
DEBUG_TEXT_XPOS = 0

MAX_FPS = 60
WIDTH   = 1280      # window width
HEIGHT  = 800       # window height

VIEW_LEFT = 250     # X position in the window
VIEW_TOP  = 150     # Y position in the window

VIEW_WIDTH  = 600   # view width
VIEW_HEIGHT = 400   # view height

SHIP_X = 450        # ie left
SHIP_Y = 730        # ie top

# -------------------------------------------------------------------------------------------------

USE_MINI_MASK = True
MAP_WIDTH = 792
MAP_HEIGHT = 1200

WHITE = (255, 255, 255)
RED   = (255, 0, 0)
LVIOLET  = (100, 0, 100)

BEAM_RADIUS = 380
BEAM_SURFACE = pygame.Surface((BEAM_RADIUS, BEAM_RADIUS))

# -------------------------------------------------------------------------------------------------

SLOW_DOWN_COEF = 2.0 # somehow the C++ version is slower with the same physics coef ?

SHIP_MASS = 0.9
SHIP_THRUST_MAX = 0.32 / SLOW_DOWN_COEF
SHIP_ANGLESTEP = 5
SHIP_ANGLE_LAND = 30
SHIP_MAX_LIVES = 100
SHIP_SPRITE_SIZE = 32

iG       = 0.07 / SLOW_DOWN_COEF
iXfrott  = 0.984
iYfrott  = 0.99
iCoeffax = 0.6
iCoeffay = 0.6
iCoeffvx = 0.6
iCoeffvy = 0.6
iCoeffimpact = 0.2

# -------------------------------------------------------------------------------------------------

PLATFORMS_1 = [ ( 464, 513, 333 ),
                ( 60, 127, 1045 ),
                ( 428, 497, 531 ),
                ( 504, 568, 985 ),
                ( 178, 241, 875 ),
                ( 8, 37, 187 ),
                ( 302, 351, 271 ),
                ( 434, 521, 835 ),
                ( 499, 586, 1165 ),
                ( 68, 145, 1181 ) ]

# -------------------------------------------------------------------------------------------------

class Sensors(object):

    @staticmethod
    def beam(window_surface, flipped_map_masks, xpos, ypos, vx, vy):

        # TODO use smaller map masks
        # TODO use only 0 to 90 degres beam mask quadran: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/examples/minimal_examples/pygame_minimal_mask_intersect_surface_line_2.py

        # in window coord, center of the player view
        ship_window_pos = (int(VIEW_WIDTH/2) + VIEW_LEFT + SHIP_SPRITE_SIZE/2 , int(VIEW_HEIGHT/2) + VIEW_TOP + SHIP_SPRITE_SIZE/2)
        #print("ship_window_pos", ship_window_pos)

        beam_surface_center = (int(BEAM_RADIUS/2), int(BEAM_RADIUS/2))

        # 30 degres step
        for angle in range(0, 359, 30):

            c = math.cos(math.radians(angle))
            s = math.sin(math.radians(angle))

            flip_x = c < 0
            flip_y = s < 0

            filpped_map_mask = flipped_map_masks[flip_x][flip_y]

            # beam final point
            x_dest = beam_surface_center[0] + BEAM_RADIUS * abs(c)
            y_dest = beam_surface_center[1] + BEAM_RADIUS * abs(s)

            BEAM_SURFACE.fill((0, 0, 0))
            BEAM_SURFACE.set_colorkey((0, 0, 0))
            pygame.draw.line(BEAM_SURFACE, WHITE, beam_surface_center, (x_dest, y_dest))
            beam_mask = pygame.mask.from_surface(BEAM_SURFACE)
            pygame.draw.circle(BEAM_SURFACE, RED, beam_surface_center, 3)

            # offset = beam mask (left/top) coordinate in the map (ie where to put our lines mask in the map)
            if flip_x:
                offset_x = MAP_WIDTH - (xpos+SHIP_SPRITE_SIZE/2) - int(BEAM_RADIUS/2)
            else:
                offset_x = xpos+SHIP_SPRITE_SIZE/2 - int(BEAM_RADIUS/2)

            if flip_y:
                offset_y = MAP_HEIGHT - (ypos+SHIP_SPRITE_SIZE/2) - int(BEAM_RADIUS/2)
            else:
                offset_y = ypos+SHIP_SPRITE_SIZE/2 - int(BEAM_RADIUS/2)

            #print("offset", offset_x, offset_y)
            hit = filpped_map_mask.overlap(beam_mask, (int(offset_x), int(offset_y)))
            #print("hit", hit)

            if hit is not None and (hit[0] != xpos+SHIP_SPRITE_SIZE/2 or hit[1] != ypos+SHIP_SPRITE_SIZE/2):
                hx = MAP_WIDTH-1 - hit[0] if flip_x else hit[0]
                hy = MAP_HEIGHT-1 - hit[1] if flip_y else hit[1]
                hit = (hx, hy)
                #print("new hit", hit)

                # go back to screen coordinates
                dx_hit = hit[0] - (xpos+SHIP_SPRITE_SIZE/2)
                dy_hit = hit[1] - (ypos+SHIP_SPRITE_SIZE/2)

                pygame.draw.line(window_surface, LVIOLET, ship_window_pos, (ship_window_pos[0] + dx_hit, ship_window_pos[1] + dy_hit)) 
                #pygame.draw.circle(map, RED, hit, 2)

            #window_surface.blit(BEAM_SURFACE, (VIEW_LEFT + VIEW_WIDTH, VIEW_TOP))
            #map.blit(BEAM_SURFACE, (int(offset_x), int(offset_y)))

    @staticmethod
    def octo(window_surface, map_mask, xpos, ypos, vx, vy, fixed_radius=False):
        # Note: with an adaptative radius the ANN would need to also know vx and vy in adition to the 8 sensors values

        if not fixed_radius:
            radius = int(SHIP_SPRITE_SIZE * 1.5)
            radius += 12 * math.sqrt( math.pow(vx, 2) + math.pow(vy, 2) )
        else:
            radius = int(SHIP_SPRITE_SIZE * 2)

        # follow V vector or fixed in the grid ?
        slx, sly  = (xpos + SHIP_SPRITE_SIZE/2) - radius, (ypos + SHIP_SPRITE_SIZE/2)
        srx, sry  = (xpos + SHIP_SPRITE_SIZE/2) + radius, (ypos + SHIP_SPRITE_SIZE/2)
        sux, suy  = (xpos + SHIP_SPRITE_SIZE/2), (ypos + SHIP_SPRITE_SIZE/2) - radius
        sdx, sdy  = (xpos + SHIP_SPRITE_SIZE/2), (ypos + SHIP_SPRITE_SIZE/2) + radius

        sulx, suly  = (xpos + SHIP_SPRITE_SIZE/2) - radius/1.4, (ypos + SHIP_SPRITE_SIZE/2) - radius/1.4
        surx, sury  = (xpos + SHIP_SPRITE_SIZE/2) + radius/1.4, (ypos + SHIP_SPRITE_SIZE/2) - radius/1.4
        sdlx, sdly  = (xpos + SHIP_SPRITE_SIZE/2) - radius/1.4, (ypos + SHIP_SPRITE_SIZE/2) + radius/1.4
        sdrx, sdry  = (xpos + SHIP_SPRITE_SIZE/2) + radius/1.4, (ypos + SHIP_SPRITE_SIZE/2) + radius/1.4

        # position for game.map for the collsion mask
        sensors_pos = [(slx, sly), (srx, sry), (sux, suy), (sdx, sdy), \
                       (sulx, suly), (surx, sury), (sdlx, sdly), (sdrx, sdry)]

        colors = []

        # collision
        for s in sensors_pos:
            try:
                color = RED if map_mask.get_at((int(s[0]), int(s[1]))) else WHITE
            # out of bound => no collision
            except:
                color = WHITE

            colors.append(color)

        # display the sensor (relocate the position for game.window)
        sensors_pos = [(SHIP_SPRITE_SIZE/2 - radius, SHIP_SPRITE_SIZE/2), (SHIP_SPRITE_SIZE/2 + radius, SHIP_SPRITE_SIZE/2), (SHIP_SPRITE_SIZE/2, SHIP_SPRITE_SIZE/2 - radius), (SHIP_SPRITE_SIZE/2, SHIP_SPRITE_SIZE/2 + radius), \
                       (SHIP_SPRITE_SIZE/2 - radius/1.4, SHIP_SPRITE_SIZE/2 - radius/1.4), (SHIP_SPRITE_SIZE/2 + radius/1.4, SHIP_SPRITE_SIZE/2 - radius/1.4), (SHIP_SPRITE_SIZE/2 - radius/1.4, SHIP_SPRITE_SIZE/2 + radius/1.4), (SHIP_SPRITE_SIZE/2 + radius/1.4, SHIP_SPRITE_SIZE/2 + radius/1.4)]

        for i, s in enumerate(sensors_pos):
            #gfxdraw.pixel(surface, int(VIEW_WIDTH/2) + VIEW_LEFT + int(s[0]) , int(VIEW_HEIGHT/2) + VIEW_TOP + int(s[1]), colors[i])
            pygame.draw.circle(window_surface, colors[i], (int(VIEW_WIDTH/2) + VIEW_LEFT + int(s[0]) , int(VIEW_HEIGHT/2) + VIEW_TOP + int(s[1])), 2, width=0)

    @staticmethod
    def circle(window_surface, map_mask, xpos, ypos, vx, vy, fixed_radius = False):

        if not fixed_radius:
            radius = int(SHIP_SPRITE_SIZE * 1.5)
            radius += 12 * math.sqrt( math.pow(vx, 2) + math.pow(vy, 2) )
        else:
            radius = int(SHIP_SPRITE_SIZE * 2)

        pygame.draw.circle(window_surface, WHITE, (int(VIEW_WIDTH/2) + VIEW_LEFT + SHIP_SPRITE_SIZE/2 , int(VIEW_HEIGHT/2) + VIEW_TOP + SHIP_SPRITE_SIZE/2), radius, width=1)

# -------------------------------------------------------------------------------------------------

class Ship(object):

    def __init__(self, xpos, ypos, lives):

        #
        self.xpos = xpos
        self.ypos = ypos
        self.xposprecise = xpos
        self.yposprecise = ypos

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.angle  = 0.0
        self.thrust = 0.0
        self.shield = False
        self.shoot  = False
        self.landed = False
        self.bounce = False
        self.explod = False

        self.lives = lives

        # sound 
        self.sound_thrust = pygame.mixer.Sound( os.path.join("assets", "default", "sfx_loop_thrust.wav") )
        self.sound_explod = pygame.mixer.Sound( os.path.join("assets", "default", "sfx_boom.wav") )
        self.sound_bounce = pygame.mixer.Sound( os.path.join("assets", "default", "sfx_rebound.wav") )

        # ship pic: 32x32, black (0,0,0) background, no alpha
        self.ship_pic = pygame.image.load( os.path.join("assets", "default", "ship1_256c.bmp") ).convert()
        self.ship_pic.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship

        self.ship_pic_thrust = pygame.image.load( os.path.join("assets", "default", "ship1_thrust_256c.bmp") ).convert()
        self.ship_pic_thrust.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship

        self.image = self.ship_pic
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, ship_controls, motion):

        if motion == "basic":

            # pic
            if ship_controls["LEFT"] or ship_controls["RIGHT"] or ship_controls["UP"] or ship_controls["DOWN"]:
                self.image = self.ship_pic_thrust
            else:
                self.image = self.ship_pic

            #
            dx = dy = 0

            if ship_controls["LEFT"]:
                dx = -1
            if ship_controls["RIGHT"]:
                dx = 1
            if ship_controls["UP"]:
                dy = -1
            if ship_controls["DOWN"]:
                dy = 1

            self.xpos += dx
            self.ypos += dy

        elif motion == "thrust":

            # pic
            if ship_controls["THRUST"]:
                self.image = self.ship_pic_thrust
            else:
                self.image = self.ship_pic

            # angle
            if ship_controls["LEFT"]:
                self.angle += SHIP_ANGLESTEP
            if ship_controls["RIGHT"]:
                self.angle -= SHIP_ANGLESTEP

            self.angle = self.angle % 360

            if ship_controls["THRUST"]:
                coef = 2
                self.xposprecise -= coef * math.cos( math.radians(90 - self.angle) )
                self.yposprecise -= coef * math.sin( math.radians(90 - self.angle) )
                
                # transfer to screen coordinates
                self.xpos = int(self.xposprecise)
                self.ypos = int(self.yposprecise)

        elif motion == "gravity":
    
            # pic
            if ship_controls["THRUST"]:
                self.image = self.ship_pic_thrust

                #self.thrust += 0.1
                #if self.thrust >= SHIP_THRUST_MAX:
                self.thrust = SHIP_THRUST_MAX

                if not pygame.mixer.get_busy():
                    self.sound_thrust.play(-1)

                self.landed = False

            else:
                self.image = self.ship_pic
                self.thrust = 0.0
                self.sound_thrust.stop()

            self.bounce = False

            if not self.landed:
                # angle
                if ship_controls["LEFT"]:
                    self.angle += SHIP_ANGLESTEP
                if ship_controls["RIGHT"]:
                    self.angle -= SHIP_ANGLESTEP

                # 
                self.angle = self.angle % 360

                # https://gafferongames.com/post/integration_basics/
                self.ax = self.thrust * -math.cos( math.radians(90 - self.angle) ) # ax = thrust * sin1
                self.ay = iG + (self.thrust * -math.sin( math.radians(90 - self.angle))) # ay = g + thrust * (-cos1)

                self.vx = self.vx + (iCoeffax * self.ax) # vx += coeffa * ax
                self.vy = self.vy + (iCoeffay * self.ay) # vy += coeffa * ay

                self.vx = self.vx * iXfrott # on freine de xfrott
                self.vy = self.vy * iYfrott # on freine de yfrott

                self.xposprecise = self.xposprecise + (iCoeffvx * self.vx) # xpos += coeffv * vx
                self.yposprecise = self.yposprecise + (iCoeffvy * self.vy) # ypos += coeffv * vy

            else:
                self.vx = 0.
                self.vy = 0.
                self.ax = 0.
                self.ay = 0.

            # transfer to screen coordinates
            self.xpos = int(self.xposprecise)
            self.ypos = int(self.yposprecise)

            # landed ?
            self.is_landed()

        #
        # rotate
        self.image_rotated = pygame.transform.rotate(self.image, self.angle)

        rect = self.image_rotated.get_rect()
        rot_xoffset = ((SHIP_SPRITE_SIZE - rect.width)/2)
        rot_yoffset = ((SHIP_SPRITE_SIZE - rect.height)/2)
        
        self.mask = pygame.mask.from_surface(self.image_rotated)

        return int(rot_xoffset), int(rot_yoffset)

    def is_landed(self):

        for plaform in PLATFORMS_1:
            xmin  = plaform[0] - (SHIP_SPRITE_SIZE - 23)
            xmax  = plaform[1] - (SHIP_SPRITE_SIZE - 9)
            yflat = plaform[2] - (SHIP_SPRITE_SIZE - 2)

            #print(self.ypos, yflat)

            if ((xmin <= self.xpos) and (self.xpos <= xmax) and
               ((self.ypos == yflat) or ((self.ypos-1) == yflat) or ((self.ypos-2) == yflat) or ((self.ypos-3) == yflat) ) and
               (self.vy > 0) and (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):

                self.vy = - self.vy / 1.2
                self.vx = self.vx / 1.1
                self.angle = 0
                self.ypos = yflat
                self.yposprecise = yflat

                if ( (-1.0/SLOW_DOWN_COEF <= self.vx) and (self.vx < 1.0/SLOW_DOWN_COEF) and (-1.0/SLOW_DOWN_COEF < self.vy) and (self.vy < 1.0/SLOW_DOWN_COEF) ):
                    self.landed = True
                    self.bounce = False
                else:
                    self.bounce = True
                    self.sound_bounce.play()

                return True

        return False

    def do_test_collision(self):
        test_it = True

        for plaform in PLATFORMS_1:
            xmin  = plaform[0] - (SHIP_SPRITE_SIZE - 23)
            xmax  = plaform[1] - (SHIP_SPRITE_SIZE - 9)
            yflat = plaform[2] - (SHIP_SPRITE_SIZE - 2)

            if ((xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos-2)==yflat) or ((self.ypos-3)==yflat))  and  (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):
                test_it = False
                break
            if (self.shield and (xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos-2)==yflat) or ((self.ypos-3)==yflat) or ((self.ypos+1)==yflat)) and  (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):
                test_it = False
                break
            if ((self.thrust) and (xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos+1)==yflat) )):
                test_it = False
                break

        return test_it

# -------------------------------------------------------------------------------------------------

class Game(object):

    def __init__(self, game_width, game_height):

        pygame.display.set_caption('Mayhem')

        # Window
        self.game_width = game_width
        self.game_height = game_height
        self.window = pygame.display.set_mode((game_width, game_height), pygame.DOUBLEBUF)

        # Background
        self.map = pygame.image.load( os.path.join("assets", "level1", "Mayhem_Level1_Map_256c.bmp") ).convert() # .convert_alpha()

        self.map.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
        #self.map_rect = self.map.get_rect()
        self.map_mask = pygame.mask.from_surface(self.map)
        self.mask_map_fx = pygame.mask.from_surface(pygame.transform.flip(self.map, True, False))
        self.mask_map_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, False, True))
        self.mask_map_fx_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, True, True))
        self.flipped_masks = [[self.map_mask, self.mask_map_fy], [self.mask_map_fx, self.mask_map_fx_fy]]

# -------------------------------------------------------------------------------------------------

class Sequence(object):
    
    def __init__(self, screen_width, screen_height, motion="gravity", sensor="", record_play=False, play_recorded=False):

        self.myfont = pygame.font.SysFont('Arial', 20)

        # screen
        self.game = Game(screen_width, screen_height)
        self.game.window.fill((0, 0, 0))

        # basic, thrust, gravity
        self.motion = motion
        self.sensor = sensor

        # record / play recorded
        self.record_play = record_play
        self.played_data = [] # [(0,0,0), (0,0,1), ...] (left, right, thrust)

        self.play_recorded = play_recorded

        if self.play_recorded:
            with open("played.dat", "rb") as f:
                self.played_data = pickle.load(f)

        #init  motion
        self.ship_controls = {}
        self.ship_controls["LEFT"]   = False
        self.ship_controls["RIGHT"]  = False
        self.ship_controls["UP"]     = False
        self.ship_controls["DOWN"]   = False
        self.ship_controls["THRUST"] = False

        # FPS
        self.clock = pygame.time.Clock()
        self.frames = 0

    def main_loop(self):

        nb_dead = 0

        # exit on Quit        
        while True:

            self.ship_1 = Ship(SHIP_X, SHIP_Y, SHIP_MAX_LIVES - nb_dead)

            # run_loop() exits when ship explods
            self.run_loop()
  
            self.ship_1.sound_thrust.stop()
            self.ship_1.sound_bounce.stop()
            self.ship_1.sound_explod.play()

            nb_dead += 1

            # record play ?
            if self.record_play:
                with open("played.dat", "wb") as f:
                    pickle.dump(self.played_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                time.sleep(0.1)
                print("Frames=", self.frames)
                print("%s seconds" % int(self.frames/MAX_FPS))
                sys.exit(0)

    def play_recorded_data(self):

        # exit ?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)

        try:
            data_i = self.played_data[self.frames]

            if data_i[0]:
                self.ship_controls["LEFT"] = True
            else:
                self.ship_controls["LEFT"] = False

            if data_i[1]:
                self.ship_controls["RIGHT"] = True
            else:
                self.ship_controls["RIGHT"] = False

            if data_i[2]:
                self.ship_controls["THRUST"] = True
            else:
                self.ship_controls["THRUST"] = False
        except:
            print("End of playback")
            print("Frames=", self.frames)
            print("%s seconds" % int(self.frames/MAX_FPS))
            sys.exit(0)

    def handle_events(self):

        for event in pygame.event.get():
            #print(event, event.type)

            if event.type == pygame.QUIT:
                sys.exit(0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)

                elif event.key == pygame.K_LEFT:
                    self.ship_controls["LEFT"] = True
                elif event.key == pygame.K_RIGHT:
                    self.ship_controls["RIGHT"] = True
                elif event.key == pygame.K_UP:
                    self.ship_controls["UP"] = True
                elif event.key == pygame.K_DOWN:
                    self.ship_controls["DOWN"] = True
                elif event.key == pygame.K_KP_PERIOD:
                    self.ship_controls["THRUST"] = True

            elif event.type == pygame.KEYUP:
    
                if event.key == pygame.K_LEFT:
                    self.ship_controls["LEFT"] = False
                elif event.key == pygame.K_RIGHT:
                    self.ship_controls["RIGHT"] = False
                elif event.key == pygame.K_UP:
                    self.ship_controls["UP"] = False
                elif event.key == pygame.K_DOWN:
                    self.ship_controls["DOWN"] = False
                elif event.key == pygame.K_KP_PERIOD:
                    self.ship_controls["THRUST"] = False

            elif event.type == pygame.JOYAXISMOTION:

                if event.axis == 0:
                    if int(event.value) == 1:
                        self.ship_controls["RIGHT"] = True
                    else:
                        self.ship_controls["RIGHT"] = False

                    if int(event.value) == -1:
                        self.ship_controls["LEFT"] = True
                    else:
                        self.ship_controls["LEFT"] = False

            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0:
                    self.ship_controls["THRUST"] = False
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.ship_controls["THRUST"] = True

            elif event.type == pygame.JOYBALLMOTION:
                print("Joystick JOYBALLMOTION pressed.")
            elif event.type == pygame.JOYHATMOTION:
                print("Joystick JOYHATMOTION pressed.")

        # record play ?
        if self.record_play:
            self.played_data.append((self.ship_controls["LEFT"], self.ship_controls["RIGHT"], self.ship_controls["THRUST"]))

    def run_loop(self):

        # Game Main Loop
        while not self.ship_1.explod:

            # pygame events
            if not self.play_recorded:
                self.handle_events()
            else:
                self.play_recorded_data()              

            # clear screen
            self.game.window.fill((0,0,0))

            # blit the map area around the ship on the screen
            sub_area = Rect(self.ship_1.xpos - VIEW_WIDTH/2, self.ship_1.ypos - VIEW_HEIGHT/2, VIEW_WIDTH, VIEW_HEIGHT)
            self.game.window.blit(self.game.map, (VIEW_LEFT, VIEW_TOP), sub_area)

            # move ship
            rot_xoffset, rot_yoffset = self.ship_1.update(self.ship_controls, self.motion)

            # blit ship
            self.game.window.blit(self.ship_1.image_rotated, (VIEW_WIDTH/2 + VIEW_LEFT + rot_xoffset, VIEW_HEIGHT/2 + VIEW_TOP + rot_yoffset))

            # mini mask: mask the size of the ship, following the ship pos in the map
            if USE_MINI_MASK:
                mini_area = Rect(self.ship_1.xpos, self.ship_1.ypos, SHIP_SPRITE_SIZE, SHIP_SPRITE_SIZE)
                mini_subsurface = self.game.map.subsurface(mini_area)
                mini_subsurface.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
                mini_mask = pygame.mask.from_surface(mini_subsurface)

                # collision
                collision_str = "No Test"
                if self.ship_1.do_test_collision():
                    offset = (rot_xoffset, rot_yoffset) # pos of the ship

                    if mini_mask.overlap(self.ship_1.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                        collision_str = "BOOM"
                        self.ship_1.explod = True
                    else:
                        collision_str = "OK"

            else:
                # collision
                collision_str = "No Test"
                if self.ship_1.do_test_collision():
                    offset = (self.ship_1.xpos + rot_xoffset, self.ship_1.ypos + rot_yoffset) # pos of the ship

                    if self.game.map_mask.overlap(self.ship_1.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                        collision_str = "BOOM"
                        self.ship_1.explod = True
                    else:
                        collision_str = "OK"

            # sensors
            if self.sensor == "octo":
                Sensors.octo(self.game.window, self.game.map_mask, self.ship_1.xpos, self.ship_1.ypos, self.ship_1.vx, self.ship_1.vy, fixed_radius=False)
            if self.sensor == "octo_fixed":
                Sensors.octo(self.game.window, self.game.map_mask, self.ship_1.xpos, self.ship_1.ypos, self.ship_1.vx, self.ship_1.vy, fixed_radius=True)
            elif self.sensor == "beam":
                Sensors.beam(self.game.window, self.game.flipped_masks, self.ship_1.xpos, self.ship_1.ypos, self.ship_1.vx, self.ship_1.vy)
            elif self.sensor == "circle":
                Sensors.circle(self.game.window, self.game.map_mask, self.ship_1.xpos, self.ship_1.ypos, self.ship_1.vx, self.ship_1.vy)

            # debug text
            if DEBUG_SCREEN:
                collision = self.myfont.render('Collision: %s' % collision_str, False, (255, 255, 255))
                self.game.window.blit(collision, (DEBUG_TEXT_XPOS + 5, 5))

                ship_pos = self.myfont.render('Ship pos: %s %s' % (self.ship_1.xpos, self.ship_1.ypos), False, (255, 255, 255))
                self.game.window.blit(ship_pos, (DEBUG_TEXT_XPOS + 5, 30))

                ship_va = self.myfont.render('Ship vx=%.2f, vy=%.2f, ax=%.2f, ay=%.2f' % (self.ship_1.vx,self.ship_1.vy, self.ship_1.ax, self.ship_1.ay), False, (255, 255, 255))
                self.game.window.blit(ship_va, (DEBUG_TEXT_XPOS + 5, 55))

                ship_angle = self.myfont.render('Ship angle: %s' % (self.ship_1.angle,), False, (255, 255, 255))
                self.game.window.blit(ship_angle, (DEBUG_TEXT_XPOS + 5, 80))

                ship_lives = self.myfont.render('Ship lives: %s' % (self.ship_1.lives,), False, (255, 255, 255))
                self.game.window.blit(ship_lives, (DEBUG_TEXT_XPOS + 5, 105))

                dt = self.myfont.render('Frames: %s' % (self.frames,), False, (255, 255, 255))
                self.game.window.blit(dt, (DEBUG_TEXT_XPOS + 5, 130))

                fps = self.myfont.render('FPS: %.2f' % self.clock.get_fps(), False, (255, 255, 255))
                self.game.window.blit(fps, (DEBUG_TEXT_XPOS + 5, 155))

            # display
            pygame.display.flip()
            self.clock.tick(MAX_FPS) # https://python-forum.io/thread-16692.html

            self.frames += 1
            #print(self.clock.get_fps())

# -------------------------------------------------------------------------------------------------

def run():
    pygame.init()
    #pygame.display.init()

    pygame.mouse.set_visible(False)
    pygame.font.init()
    pygame.mixer.init() # frequency=22050

    #pygame.event.set_blocked((MOUSEMOTION, MOUSEBUTTONUP, MOUSEBUTTONDOWN))

    # joystick
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    print("joystick_count", joystick_count)

    for i in range(joystick_count):
        j = pygame.joystick.Joystick(i)
        j.init()

    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--motion', help='How the ship moves', action="store", default='gravity', choices=("basic", "thrust", "gravity"))
    parser.add_argument('-r', '--record_play', help='', action="store_true", default=False)
    parser.add_argument('-pr', '--play_recorded', help='', action="store_true", default=False)
    parser.add_argument('-s', '--sensor', help='', action="store", default="", choices=("octo", "octo_fixed", "beam", "circle", ""))

    result = parser.parse_args()
    args = dict(result._get_kwargs())

    print("Args", args)

    # env
    seq = Sequence(WIDTH, HEIGHT, motion=args["motion"], sensor=args["sensor"], record_play=args["record_play"], play_recorded=args["play_recorded"])
    seq.main_loop()

if __name__ == '__main__':
    run()
