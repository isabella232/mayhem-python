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

"""
Usage example:

python mayhem.py --width=1500 --height=900 --nb_player=2 --sensor=beam -rm=game
python mayhem.py --width=1500 --height=900 --nb_player=1 --sensor=beam -rm=training

python3 mayhem.py --sensor=beam --motion=gravity
python3 mayhem.py --sensor=octo --motion=gravity
python3 mayhem.py --sensor=beam --motion=thrust
python3 mayhem.py --motion=thrust
python3 mayhem.py -r=played1.dat --motion=gravity
python3 mayhem.py -pr=played1.dat --motion=gravity
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
# General

DEBUG_SCREEN = False # print debug info on the screen
DEBUG_TEXT_XPOS = 0

MAX_FPS = 60

MAP_WIDTH  = 792
MAP_HEIGHT = 1200

WHITE    = (255, 255, 255)
RED      = (255, 0, 0)
LVIOLET  = (128, 0, 128)

# -------------------------------------------------------------------------------------------------
# Player views

SHIP1_X = 473        # ie left
SHIP1_Y = 303        # ie top

SHIP2_X = 520        # ie left
SHIP2_Y = 955        # ie top

SHIP3_X = 75         # ie left
SHIP3_Y = 1015       # ie top

SHIP4_X = 451        # ie left
SHIP4_Y = 501        # ie top

USE_MINI_MASK = True # mask the size of the ship (instead of the player view size)

# -------------------------------------------------------------------------------------------------
# Sensors

BEAM_AMGLE_STEP = 30
BEAM_RADIUS = 400
BEAM_SURFACE = pygame.Surface((BEAM_RADIUS, BEAM_RADIUS))

# -------------------------------------------------------------------------------------------------
# SHIP dynamics

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
iCoeffimpact = 0.02
MAX_SHOOT = 20

# -------------------------------------------------------------------------------------------------
# Levels / controls

CURRENT_LEVEL = 1

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

SHIP_1_KEYS = {"left":pygame.K_LEFT, "right":pygame.K_RIGHT, "up":pygame.K_UP, "down":pygame.K_DOWN, \
               "thrust":pygame.K_KP_PERIOD, "shoot":pygame.K_KP_ENTER, "shield":pygame.K_KP0}
SHIP_1_JOY  = 0 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

SHIP_2_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "up":pygame.K_UP, "down":pygame.K_DOWN, \
               "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_2_JOY  = 1 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

SHIP_3_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "up":pygame.K_UP, "down":pygame.K_DOWN, \
               "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_3_JOY  = 2 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

SHIP_4_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "up":pygame.K_UP, "down":pygame.K_DOWN, \
               "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_4_JOY  = 0 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

# -------------------------------------------------------------------------------------------------
# Assets

MAP_1 = os.path.join("assets", "level1", "Mayhem_Level1_Map_256c.bmp")

SOUND_THURST  = os.path.join("assets", "default", "sfx_loop_thrust.wav")
SOUND_EXPLOD  = os.path.join("assets", "default", "sfx_boom.wav")
SOUND_BOUNCE  = os.path.join("assets", "default", "sfx_rebound.wav")
SOUND_SHOOT   = os.path.join("assets", "default", "sfx_shoot.wav")
SOUND_SHIELD  = os.path.join("assets", "default", "sfx_loop_shield.wav")

SHIP_1_PIC        = os.path.join("assets", "default", "ship1_256c.bmp")
SHIP_1_PIC_THRUST = os.path.join("assets", "default", "ship1_thrust_256c.bmp")
SHIP_1_PIC_SHIELD = os.path.join("assets", "default", "ship1_shield_256c.bmp")

SHIP_2_PIC        = os.path.join("assets", "default", "ship2_256c.bmp")
SHIP_2_PIC_THRUST = os.path.join("assets", "default", "ship2_thrust_256c.bmp")
SHIP_2_PIC_SHIELD = os.path.join("assets", "default", "ship2_shield_256c.bmp")

SHIP_3_PIC        = os.path.join("assets", "default", "ship3_256c.bmp")
SHIP_3_PIC_THRUST = os.path.join("assets", "default", "ship3_thrust_256c.bmp")
SHIP_3_PIC_SHIELD = os.path.join("assets", "default", "ship3_shield_256c.bmp")

SHIP_4_PIC        = os.path.join("assets", "default", "ship4_256c.bmp")
SHIP_4_PIC_THRUST = os.path.join("assets", "default", "ship4_thrust_256c.bmp")
SHIP_4_PIC_SHIELD = os.path.join("assets", "default", "ship4_shield_256c.bmp")

# -------------------------------------------------------------------------------------------------

class Sensors():

    @staticmethod
    def beam(game, ship):
        # TODO use smaller map masks
        # TODO use only 0 to 90 degres beam mask quadran: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/examples/minimal_examples/pygame_minimal_mask_intersect_surface_line_2.py

        # clipping translation for window coordinates
        rx = ship.xpos - ship.view_width/2
        ry = ship.ypos - ship.view_height/2

        dx = 0
        dy = 0

        if rx < 0:
            dx = rx
        elif rx > (MAP_WIDTH - ship.view_width):
            dx = rx - (MAP_WIDTH - ship.view_width)
        if ry < 0:
            dy = ry
        elif ry > (MAP_HEIGHT - ship.view_height):
            dy = ry - (MAP_HEIGHT - ship.view_height)

        #sub_area1 = Rect(rx, ry, ship.view_width, ship.view_height)
        #self.game.window.blit(self.game.map_buffer, (ship.view_left, ship.view_top), sub_area1)

        # in window coord, center of the player view
        ship_window_pos = (int(ship.view_width/2) + ship.view_left + SHIP_SPRITE_SIZE/2 + dx, int(ship.view_height/2) + ship.view_top + SHIP_SPRITE_SIZE/2 + dy)
        #print("ship_window_pos", ship_window_pos)

        beam_surface_center = (int(BEAM_RADIUS/2), int(BEAM_RADIUS/2))

        # 30 degres step
        for angle in range(0, 359, BEAM_AMGLE_STEP):

            c = math.cos(math.radians(angle))
            s = math.sin(math.radians(angle))

            flip_x = c < 0
            flip_y = s < 0

            filpped_map_mask = game.flipped_masks_map_buffer[flip_x][flip_y]

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
                offset_x = MAP_WIDTH - (ship.xpos+SHIP_SPRITE_SIZE/2) - int(BEAM_RADIUS/2)
            else:
                offset_x = ship.xpos+SHIP_SPRITE_SIZE/2 - int(BEAM_RADIUS/2)

            if flip_y:
                offset_y = MAP_HEIGHT - (ship.ypos+SHIP_SPRITE_SIZE/2) - int(BEAM_RADIUS/2)
            else:
                offset_y = ship.ypos+SHIP_SPRITE_SIZE/2 - int(BEAM_RADIUS/2)

            #print("offset", offset_x, offset_y)
            hit = filpped_map_mask.overlap(beam_mask, (int(offset_x), int(offset_y)))
            #print("hit", hit)

            if hit is not None and (hit[0] != ship.xpos+SHIP_SPRITE_SIZE/2 or hit[1] != ship.ypos+SHIP_SPRITE_SIZE/2):
                hx = MAP_WIDTH-1 - hit[0] if flip_x else hit[0]
                hy = MAP_HEIGHT-1 - hit[1] if flip_y else hit[1]
                hit = (hx, hy)
                #print("new hit", hit)

                # go back to screen coordinates
                dx_hit = hit[0] - (ship.xpos+SHIP_SPRITE_SIZE/2)
                dy_hit = hit[1] - (ship.ypos+SHIP_SPRITE_SIZE/2)

                pygame.draw.line(game.window, LVIOLET, ship_window_pos, (ship_window_pos[0] + dx_hit, ship_window_pos[1] + dy_hit))
                #pygame.draw.circle(map, RED, hit, 2)

                # Note: this is the distance from the center of the ship, not the borders
                dist_wall = math.sqrt(dx_hit**2 + dy_hit**2)
                # so remove
                dist_wall -= (SHIP_SPRITE_SIZE/2 - 1)

            # Not hit: too far
            else:
                dist_wall = (BEAM_RADIUS/2) * math.sqrt(2)

            if dist_wall < 0:
                dist_wall = 0

            #print("Sensor for angle=%s, dist wall=%.2f" % (str(angle), dist_wall))

            #game.window.blit(BEAM_SURFACE, (ship.view_left + ship.view_width, ship.view_top))
            #map.blit(BEAM_SURFACE, (int(offset_x), int(offset_y)))

    @staticmethod
    def octo(game, ship, fixed_radius=False):

        # Note: with an adaptative radius the ANN would need to also know vx and vy in adition to the 8 sensors values
        if not fixed_radius:
            radius = int(SHIP_SPRITE_SIZE * 1.5)
            radius += 12 * math.sqrt( math.pow(ship.vx, 2) + math.pow(ship.vy, 2) )
        else:
            radius = int(SHIP_SPRITE_SIZE * 2)

        # follow V vector or fixed in the grid ?
        slx, sly  = (ship.xpos + SHIP_SPRITE_SIZE/2) - radius, (ship.ypos + SHIP_SPRITE_SIZE/2)
        srx, sry  = (ship.xpos + SHIP_SPRITE_SIZE/2) + radius, (ship.ypos + SHIP_SPRITE_SIZE/2)
        sux, suy  = (ship.xpos + SHIP_SPRITE_SIZE/2), (ship.ypos + SHIP_SPRITE_SIZE/2) - radius
        sdx, sdy  = (ship.xpos + SHIP_SPRITE_SIZE/2), (ship.ypos + SHIP_SPRITE_SIZE/2) + radius

        sulx, suly  = (ship.xpos + SHIP_SPRITE_SIZE/2) - radius/1.4, (ship.ypos + SHIP_SPRITE_SIZE/2) - radius/1.4
        surx, sury  = (ship.xpos + SHIP_SPRITE_SIZE/2) + radius/1.4, (ship.ypos + SHIP_SPRITE_SIZE/2) - radius/1.4
        sdlx, sdly  = (ship.xpos + SHIP_SPRITE_SIZE/2) - radius/1.4, (ship.ypos + SHIP_SPRITE_SIZE/2) + radius/1.4
        sdrx, sdry  = (ship.xpos + SHIP_SPRITE_SIZE/2) + radius/1.4, (ship.ypos + SHIP_SPRITE_SIZE/2) + radius/1.4

        # position for game.map for the collsion mask
        sensors_pos = [(slx, sly), (srx, sry), (sux, suy), (sdx, sdy), \
                       (sulx, suly), (surx, sury), (sdlx, sdly), (sdrx, sdry)]

        colors = []

        # collision
        for s in sensors_pos:
            try:
                color = RED if game.map_buffer_mask.get_at((int(s[0]), int(s[1]))) else WHITE
            # out of bound => no collision
            except:
                color = WHITE

            colors.append(color)

        # display the sensor (relocate the position for game.window)
        sensors_pos = [(SHIP_SPRITE_SIZE/2 - radius, SHIP_SPRITE_SIZE/2), (SHIP_SPRITE_SIZE/2 + radius, SHIP_SPRITE_SIZE/2), (SHIP_SPRITE_SIZE/2, SHIP_SPRITE_SIZE/2 - radius), (SHIP_SPRITE_SIZE/2, SHIP_SPRITE_SIZE/2 + radius), \
                       (SHIP_SPRITE_SIZE/2 - radius/1.4, SHIP_SPRITE_SIZE/2 - radius/1.4), (SHIP_SPRITE_SIZE/2 + radius/1.4, SHIP_SPRITE_SIZE/2 - radius/1.4), (SHIP_SPRITE_SIZE/2 - radius/1.4, SHIP_SPRITE_SIZE/2 + radius/1.4), (SHIP_SPRITE_SIZE/2 + radius/1.4, SHIP_SPRITE_SIZE/2 + radius/1.4)]

        # clipping translation for window coordinates
        rx = ship.xpos - ship.view_width/2
        ry = ship.ypos - ship.view_height/2

        dx = 0
        dy = 0

        if rx < 0:
            dx = rx
        elif rx > (MAP_WIDTH - ship.view_width):
            dx = rx - (MAP_WIDTH - ship.view_width)
        if ry < 0:
            dy = ry
        elif ry > (MAP_HEIGHT - ship.view_height):
            dy = ry - (MAP_HEIGHT - ship.view_height)

        for i, s in enumerate(sensors_pos):
            #gfxdraw.pixel(surface, int(ship.view_width/2) + ship.view_left + int(s[0]) , int(ship.view_height/2) + ship.view_top + int(s[1]), colors[i])
            pygame.draw.circle(game.window, colors[i], (int(ship.view_width/2) + ship.view_left + int(s[0] + dx) , int(ship.view_height/2) + ship.view_top + int(s[1]) + dy), 2, width=0)

# -------------------------------------------------------------------------------------------------

class Shot():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.xposprecise = 0
        self.yposprecise = 0
        self.dx = 0
        self.dy = 0

# -------------------------------------------------------------------------------------------------

class Ship():

    def __init__(self, screen_width, screen_height, ship_number, nb_player, xpos, ypos, ship_pic, ship_pic_thrust, ship_pic_shield, keys_mapping, joystick_number, lives):

        # TODO
        self.view_width = 600
        self.view_height = 400

        margin_size = 0
        w_percent = 1.0
        h_percent = 1.0

        self.view_width  = int((screen_width * w_percent) / 2)
        self.view_height = int((screen_height * h_percent) / 2)

        if ship_number == 1:
            self.view_left = margin_size
            self.view_top = margin_size

        elif ship_number == 2:
            self.view_left = margin_size + self.view_width + margin_size
            self.view_top = margin_size

        elif ship_number == 3:
            self.view_left = margin_size
            self.view_top = margin_size + self.view_height + margin_size

        elif ship_number == 4:
            self.view_left = margin_size + self.view_width + margin_size
            self.view_top = margin_size + self.view_height + margin_size

        self.init_xpos = xpos
        self.init_ypos = ypos
        
        self.xpos = xpos
        self.ypos = ypos
        self.xposprecise = xpos
        self.yposprecise = ypos

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.impactx = 0.0
        self.impacty = 0.0

        self.angle  = 0.0
        self.thrust = 0.0
        self.shield = False
        self.shoot  = False
        self.shoot_delay = False
        self.landed = False
        self.bounce = False
        self.explod = False

        self.lives = lives
        self.shots = []

        # sound
        self.sound_thrust = pygame.mixer.Sound(SOUND_THURST)
        self.sound_explod = pygame.mixer.Sound(SOUND_EXPLOD)
        self.sound_bounce = pygame.mixer.Sound(SOUND_BOUNCE)
        self.sound_shoot  = pygame.mixer.Sound(SOUND_SHOOT)
        self.sound_shield = pygame.mixer.Sound(SOUND_SHIELD)

        # ship pic: 32x32, black (0,0,0) background, no alpha
        self.ship_pic = pygame.image.load(ship_pic).convert()
        self.ship_pic.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship
        self.ship_pic_thrust = pygame.image.load(ship_pic_thrust).convert()
        self.ship_pic_thrust.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship
        self.ship_pic_shield = pygame.image.load(ship_pic_shield).convert()
        self.ship_pic_shield.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship

        self.image = self.ship_pic
        self.mask = pygame.mask.from_surface(self.image)

        self.keys_mapping = keys_mapping
        self.joystick_number = joystick_number

    def update(self, sequence):

        # normal play
        if not sequence.play_recorded:
            keys = pygame.key.get_pressed()

            left_pressed   = keys[self.keys_mapping["left"]]
            right_pressed  = keys[self.keys_mapping["right"]]
            up_pressed     = keys[self.keys_mapping["up"]]
            down_pressed   = keys[self.keys_mapping["down"]]
            thrust_pressed = keys[self.keys_mapping["thrust"]]
            shoot_pressed  = keys[self.keys_mapping["shoot"]]
            shield_pressed = keys[self.keys_mapping["shield"]]

            if self.joystick_number:
                try:
                    if pygame.joystick.Joystick(self.joystick_number-1).get_button(0):
                        thrust_pressed = True
                    else:
                        thrust_pressed = False

                    if pygame.joystick.Joystick(self.joystick_number-1).get_button(5):
                        shoot_pressed = True
                    else:
                        shoot_pressed = False

                    if pygame.joystick.Joystick(self.joystick_number-1).get_button(1):
                        shield_pressed = True
                    else:
                        shield_pressed = False

                    horizontal_axis = pygame.joystick.Joystick(self.joystick_number-1).get_axis(0)

                    if int(round(horizontal_axis)) == 1:
                        right_pressed = True
                    else:
                        right_pressed = False

                    if int(round(horizontal_axis)) == -1:
                        left_pressed = True
                    else:
                        left_pressed = False
                except:
                    pass

            # record play ?
            if sequence.record_play:
                sequence.played_data.append((left_pressed, right_pressed, thrust_pressed, shield_pressed, shoot_pressed))

        # play recorded sequence
        else:
            try:
                data_i = sequence.played_data[sequence.frames]

                left_pressed   = True if data_i[0] else False
                right_pressed  = True if data_i[1] else False
                thrust_pressed = True if data_i[2] else False
                shield_pressed = True if data_i[3] else False
                shoot_pressed  = True if data_i[4] else False
            except:
                print("End of playback")
                print("Frames=", sequence.frames)
                print("%s seconds" % int(sequence.frames/MAX_FPS))
                sys.exit(0)

        if sequence.motion == "basic":

            # pic
            if left_pressed or right_pressed or up_pressed or down_pressed:
                self.image = self.ship_pic_thrust
            else:
                self.image = self.ship_pic

            #
            dx = dy = 0

            if left_pressed:
                dx = -1
            if right_pressed:
                dx = 1
            if up_pressed:
                dy = -1
            if down_pressed:
                dy = 1

            self.xpos += dx
            self.ypos += dy

        elif sequence.motion == "thrust":

            # pic
            if thrust_pressed:
                self.image = self.ship_pic_thrust
            else:
                self.image = self.ship_pic

            # angle
            if left_pressed:
                self.angle += SHIP_ANGLESTEP
            if right_pressed:
                self.angle -= SHIP_ANGLESTEP

            self.angle = self.angle % 360

            if thrust_pressed:
                coef = 2
                self.xposprecise -= coef * math.cos( math.radians(90 - self.angle) )
                self.yposprecise -= coef * math.sin( math.radians(90 - self.angle) )
                
                # transfer to screen coordinates
                self.xpos = int(self.xposprecise)
                self.ypos = int(self.yposprecise)

        elif sequence.motion == "gravity":
    
            self.image = self.ship_pic
            self.thrust = 0.0
            self.shield = False

            # shield
            if shield_pressed:
                self.image = self.ship_pic_shield
                self.shield = True
                self.sound_thrust.stop()

                if not pygame.mixer.get_busy():
                    self.sound_shield.play(-1)
            else:
                self.shield = False
                self.sound_shield.stop()

                # thrust
                if thrust_pressed:
                    self.image = self.ship_pic_thrust

                    #self.thrust += 0.1
                    #if self.thrust >= SHIP_THRUST_MAX:
                    self.thrust = SHIP_THRUST_MAX

                    if not pygame.mixer.get_busy():
                        self.sound_thrust.play(-1)

                    self.landed = False

                else:
                    self.thrust = 0.0
                    self.sound_thrust.stop()

            # shoot delay
            if shoot_pressed and not self.shoot:
                self.shoot_delay = True
            else:
                self.shoot_delay = False

            # shoot
            if shoot_pressed:
                self.shoot = True

                if self.shoot_delay:
                    if len(self.shots) < MAX_SHOOT:
                        if not pygame.mixer.get_busy():
                            self.sound_shoot.play()

                        self.add_shots()
            else:
                self.shoot = False
                self.sound_shoot.stop()

            #
            self.bounce = False

            if not self.landed:
                # angle
                if left_pressed:
                    self.angle += SHIP_ANGLESTEP
                if right_pressed:
                    self.angle -= SHIP_ANGLESTEP

                # 
                self.angle = self.angle % 360

                # https://gafferongames.com/post/integration_basics/
                self.ax = self.thrust * -math.cos( math.radians(90 - self.angle) ) # ax = thrust * sin1
                self.ay = iG + (self.thrust * -math.sin( math.radians(90 - self.angle))) # ay = g + thrust * (-cos1)

                # shoot when shield is on
                if self.impactx or self.impacty:
                    self.ax += iCoeffimpact * self.impactx
                    self.ay += iCoeffimpact * self.impacty
                    self.impactx = 0.
                    self.impacty = 0.

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
        self.mask = pygame.mask.from_surface(self.image_rotated)

        rect = self.image_rotated.get_rect()
        self.rot_xoffset = int( ((SHIP_SPRITE_SIZE - rect.width)/2) )  # used in draw() and collide_map()
        self.rot_yoffset = int( ((SHIP_SPRITE_SIZE - rect.height)/2) ) # used in draw() and collide_map()

    def plot_shots(self, map_buffer):
        for shot in list(self.shots): # copy of self.shots
            shot.xposprecise += shot.dx
            shot.yposprecise += shot.dy
            shot.x = int(shot.xposprecise)
            shot.y = int(shot.yposprecise)

            try:
                c = map_buffer.get_at((int(shot.x), int(shot.y)))
                if (c.r != 0) or (c.g != 0) or (c.b != 0):
                    self.shots.remove(shot)

                #gfxdraw.pixel(map_buffer, int(shot.x) , int(shot.y), WHITE)
                pygame.draw.circle(map_buffer, WHITE, (int(shot.x) , int(shot.y)), 1)
                #pygame.draw.line(map_buffer, WHITE, (int(self.xpos + SHIP_SPRITE_SIZE/2), int(self.ypos + SHIP_SPRITE_SIZE/2)), (int(shot.x), int(shot.y)))

            # out of surface
            except IndexError:
                self.shots.remove(shot)

        if 0:
            for i in range(len(self.shots)):
                try:
                    shot1 = self.shots[i]
                    shot2 = self.shots[i+1]
                    pygame.draw.line(map_buffer, WHITE, (int(shot1.x), int(shot1.y)), (int(shot2.x), int(shot2.y)))
                except IndexError:
                    pass

    def add_shots(self):
        shot = Shot()

        shot.x = (self.xpos+15) + 18 * -math.cos(math.radians(90 - self.angle))
        shot.y = (self.ypos+16) + 18 * -math.sin(math.radians(90 - self.angle))
        shot.xposprecise = shot.x
        shot.yposprecise = shot.y
        shot.dx = 5.1 * -math.cos(math.radians(90 - self.angle))
        shot.dy = 5.1 * -math.sin(math.radians(90 - self.angle))
        shot.dx += self.vx / 3.5
        shot.dy += self.vy / 3.5

        self.shots.append(shot)

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

            #if ((xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos-2)==yflat) or ((self.ypos-3)==yflat))  and  (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):
            #    test_it = False
            #    break
            if (self.shield and (xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos-2)==yflat) or ((self.ypos-3)==yflat) or ((self.ypos+1)==yflat)) and  (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):
                test_it = False
                break
            if ((self.thrust) and (xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos+1)==yflat) )):
                test_it = False
                break

        return test_it

    def draw(self, map_buffer):
        #game_window.blit(self.image_rotated, (self.view_width/2 + self.view_left + self.rot_xoffset, self.view_height/2 + self.view_top + self.rot_yoffset))
        map_buffer.blit(self.image_rotated, (self.xpos + self.rot_xoffset, self.ypos + self.rot_yoffset))

    def reset(self):
        self.xpos = self.init_xpos
        self.ypos = self.init_ypos

        self.xposprecise = self.xpos
        self.yposprecise = self.ypos

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.impactx = 0.0
        self.impacty = 0.0

        self.angle  = 0.0
        self.thrust = 0.0
        self.shield = False
        self.shoot  = False
        self.shoot_delay = False
        self.landed = False
        self.bounce = False
        self.explod = False

        self.lives -= 1

        self.sound_thrust.stop()
        self.sound_shoot.stop()
        self.sound_shield.stop()
        self.sound_bounce.stop()
        self.sound_explod.play()

    def collide_map(self, map_buffer, map_buffer_mask):

        # ship size mask
        if USE_MINI_MASK:
            mini_area = Rect(self.xpos, self.ypos, SHIP_SPRITE_SIZE, SHIP_SPRITE_SIZE)
            mini_subsurface = map_buffer.subsurface(mini_area)
            mini_subsurface.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
            mini_mask = pygame.mask.from_surface(mini_subsurface)

            if self.do_test_collision():
                offset = (self.rot_xoffset, self.rot_yoffset) # pos of the ship

                if mini_mask.overlap(self.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                    self.explod = True

        # player view size mask
        else:
            if self.do_test_collision():
                offset = (self.xpos + self.rot_xoffset, self.ypos + self.rot_yoffset) # pos of the ship

                if map_buffer_mask.overlap(self.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                    self.explod = True

    # 
    def collide_ship(self, ships):
        for ship in ships:
            if self != ship:
                offset = ((ship.xpos - self.xpos), (ship.ypos - self.ypos))
                if self.mask.overlap(ship.mask, offset):
                    self.explod = True
                    ship.explod = True

    # 
    def collide_shots(self, ships):
        for ship in ships:
            if self != ship:
                for shot in self.shots:
                    try:
                        if ship.mask.get_at((shot.x - ship.xpos, shot.y - ship.ypos)):
                            if not ship.shield:
                                ship.explod = True
                                return
                            else:
                                ship.impactx = shot.dx
                                ship.impacty = shot.dy
                    # out of ship mask => no collision
                    except IndexError:
                        pass

# -------------------------------------------------------------------------------------------------

class Game():

    def __init__(self, game_width, game_height):

        pygame.display.set_caption('Mayhem')

        # Window
        self.game_width = game_width
        self.game_height = game_height
        flags = pygame.DOUBLEBUF | pygame.SCALED | pygame.NOFRAME # | pygame.FULLSCREEN 
        self.window = pygame.display.set_mode((game_width, game_height), flags)

        # Background
        self.map = pygame.image.load(MAP_1).convert() # .convert_alpha()
        #self.map.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
        #self.map_rect = self.map.get_rect()
        #self.map_mask = pygame.mask.from_surface(self.map)
        #self.mask_map_fx = pygame.mask.from_surface(pygame.transform.flip(self.map, True, False))
        #self.mask_map_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, False, True))
        #self.mask_map_fx_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, True, True))
        #self.flipped_masks = [[self.map_mask, self.mask_map_fy], [self.mask_map_fx, self.mask_map_fx_fy]]

        self.map_buffer = self.map.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask = pygame.mask.from_surface(self.map_buffer)
        self.mask_map_buffer_fx = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer, True, False))
        self.mask_map_buffer_fy = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer, False, True))
        self.mask_map_buffer_fx_fy = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer, True, True))
        self.flipped_masks_map_buffer = [[self.map_buffer_mask, self.mask_map_buffer_fy], [self.mask_map_buffer_fx, self.mask_map_buffer_fx_fy]]

# -------------------------------------------------------------------------------------------------

class Sequence():
    
    def __init__(self, screen_width, screen_height, nb_player, mode="game", motion="gravity", sensor="", record_play="", play_recorded=""):

        self.myfont = pygame.font.SysFont('Arial', 20)

        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.nb_player = nb_player

        # screen
        self.game = Game(screen_width, screen_height)
        self.game.window.fill((0, 0, 0))

        self.mode   = mode # training or game
        self.motion = motion # basic, thrust, gravity
        self.sensor = sensor

        # record / play recorded
        self.record_play = record_play
        self.played_data = [] # [(0,0,0), (0,0,1), ...] (left, right, thrust)

        self.play_recorded = play_recorded

        if self.play_recorded:
            with open(self.play_recorded, "rb") as f:
                self.played_data = pickle.load(f)

        # FPS
        self.clock = pygame.time.Clock()
        self.paused = False
        self.frames = 0

    def main_loop(self):

        nb_dead = 0

        # exit on Quit
        while True:

            self.ships = []

            self.ship_1 = Ship(self.screen_width, self.screen_height, 1, self.nb_player, SHIP1_X, SHIP1_Y, \
                               SHIP_1_PIC, SHIP_1_PIC_THRUST, SHIP_1_PIC_SHIELD, SHIP_1_KEYS, SHIP_1_JOY, SHIP_MAX_LIVES - nb_dead)

            self.ship_2 = Ship(self.screen_width, self.screen_height, 2, self.nb_player, SHIP2_X, SHIP2_Y, \
                               SHIP_2_PIC, SHIP_2_PIC_THRUST, SHIP_2_PIC_SHIELD, SHIP_2_KEYS, SHIP_2_JOY, SHIP_MAX_LIVES - nb_dead)

            self.ship_3 = Ship(self.screen_width, self.screen_height, 3, self.nb_player, SHIP3_X, SHIP3_Y, \
                               SHIP_3_PIC, SHIP_3_PIC_THRUST, SHIP_3_PIC_SHIELD, SHIP_3_KEYS, SHIP_3_JOY, SHIP_MAX_LIVES - nb_dead)

            self.ship_4 = Ship(self.screen_width, self.screen_height, 4, self.nb_player, SHIP4_X, SHIP4_Y, \
                               SHIP_4_PIC, SHIP_4_PIC_THRUST, SHIP_4_PIC_SHIELD, SHIP_4_KEYS, SHIP_4_JOY, SHIP_MAX_LIVES - nb_dead)

            self.ships.append(self.ship_1)
            if self.nb_player >= 2:
                self.ships.append(self.ship_2)
            if self.nb_player >= 3:
                self.ships.append(self.ship_3)
            if self.nb_player >= 4:
                self.ships.append(self.ship_4)

            # ----- training_loop() exits when ship explods
            if self.mode == "training":
                self.ship_1 = Ship(self.screen_width, self.screen_height, 1, 1, 430, 730, \
                                SHIP_1_PIC, SHIP_1_PIC_THRUST, SHIP_1_PIC_SHIELD, SHIP_1_KEYS, SHIP_1_JOY, SHIP_MAX_LIVES - nb_dead)

                self.training_loop(nb_dead)

            elif self.mode == "game":
                self.game_multiplayer_loop()

            for ship in self.ships:
                ship.sound_thrust.stop()
                ship.sound_bounce.stop()
                ship.sound_shield.stop()
                ship.sound_shoot.stop()
                ship.sound_explod.play()

            nb_dead += 1

            # record play ?
            self.record_it()

    def record_it(self):
        if self.record_play:
            with open(self.record_play, "wb") as f:
                pickle.dump(self.played_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(0.1)
            print("Frames=", self.frames)
            print("%s seconds" % int(self.frames/MAX_FPS))
            sys.exit(0)

    def game_multiplayer_loop(self):

        # Game Main Loop
        while True:

            # pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.record_it()
                    sys.exit(0)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.record_it()
                        sys.exit(0)
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused

            if not self.paused:

                # clear screen
                self.game.window.fill((0,0,0))

                # map copy (TODO reduce)
                self.game.map_buffer.blit(self.game.map, (0, 0))

                # update ship pos
                for ship in self.ships:
                    ship.update(self)

                # collide_map and ship tp ship
                for ship in self.ships:
                    ship.collide_map(self.game.map_buffer, self.game.map_buffer_mask)

                for ship in self.ships:
                    ship.collide_ship(self.ships)
                    
                for ship in self.ships:
                    ship.plot_shots(self.game.map_buffer)

                for ship in self.ships:
                    ship.collide_shots(self.ships)

                # blit ship in the map
                for ship in self.ships:
                    ship.draw(self.game.map_buffer)

                for ship in self.ships:

                    # clipping to avoid black when the ship is close to the edges
                    rx = ship.xpos - ship.view_width/2
                    ry = ship.ypos - ship.view_height/2
                    if rx < 0:
                        rx = 0
                    elif rx > (MAP_WIDTH - ship.view_width):
                        rx = (MAP_WIDTH - ship.view_width)
                    if ry < 0:
                        ry = 0
                    elif ry > (MAP_HEIGHT - ship.view_height):
                        ry = (MAP_HEIGHT - ship.view_height)

                    # blit the map area around the ship on the screen
                    sub_area1 = Rect(rx, ry, ship.view_width, ship.view_height)
                    self.game.window.blit(self.game.map_buffer, (ship.view_left, ship.view_top), sub_area1)

                # sensors
                if self.sensor == "octo":
                    for ship in self.ships:
                        Sensors.octo(self.game, ship, fixed_radius=False)
                if self.sensor == "octo_fixed":
                    for ship in self.ships:
                        Sensors.octo(self.game, ship, fixed_radius=True)
                elif self.sensor == "beam":
                    for ship in self.ships:
                        Sensors.beam(self.game, ship)

                for ship in self.ships:
                    if ship.explod:
                        ship.reset()

                # debug on screen
                self.screen_print_info()

                cv = (225, 225, 225)
                pygame.draw.line( self.game.window, cv, (0, int(self.screen_height/2)), (self.screen_width, int(self.screen_height/2)) )
                pygame.draw.line( self.game.window, cv, (int(self.screen_width/2), 0), (int(self.screen_width/2), (self.screen_height)) )

                # display
                pygame.display.flip()
                self.frames += 1

            self.clock.tick(MAX_FPS) # https://python-forum.io/thread-16692.html

            #print(self.clock.get_fps())

    def training_loop(self, nb_dead):

        # Game Main Loop
        while not self.ship_1.explod:

            # clear screen
            self.game.window.fill((0,0,0))

            # pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit(0)


            # map copy (TODO reduce)
            self.game.map_buffer.blit(self.game.map, (0, 0))

            self.ship_1.update(self)

            # collision
            self.ship_1.collide_map(self.game.map_buffer, self.game.map_buffer_mask)

            # blit ship in the map
            self.ship_1.draw(self.game.map_buffer)

            # clipping to avoid black when the ship is close to the edges
            rx = self.ship_1.xpos - self.ship_1.view_width/2
            ry = self.ship_1.ypos - self.ship_1.view_height/2
            if rx < 0:
                rx = 0
            elif rx > (MAP_WIDTH - self.ship_1.view_width):
                rx = (MAP_WIDTH - self.ship_1.view_width)
            if ry < 0:
                ry = 0
            elif ry > (MAP_HEIGHT - self.ship_1.view_height):
                ry = (MAP_HEIGHT - self.ship_1.view_height)

            # blit the map area around the ship on the screen
            sub_area1 = Rect(rx, ry, self.ship_1.view_width, self.ship_1.view_height)
            self.game.window.blit(self.game.map_buffer, (self.ship_1.view_left, self.ship_1.view_top), sub_area1)

            # sensors
            if self.sensor == "octo":
                Sensors.octo(self.game, self.ship_1, fixed_radius=False)
            if self.sensor == "octo_fixed":
                Sensors.octo(self.game, self.ship_1, fixed_radius=True)
            elif self.sensor == "beam":
                Sensors.beam(self.game, self.ship_1)

            # debug on screen
            self.screen_print_info()

            # display
            pygame.display.flip()
            self.clock.tick(MAX_FPS) # https://python-forum.io/thread-16692.html

            self.frames += 1
            #print(self.clock.get_fps())

    def screen_print_info(self):
        # debug text
        if DEBUG_SCREEN:
            ship_pos = self.myfont.render('Pos: %s %s' % (self.ship_1.xpos, self.ship_1.ypos), False, (255, 255, 255))
            self.game.window.blit(ship_pos, (DEBUG_TEXT_XPOS + 5, 30))

            ship_va = self.myfont.render('vx=%.2f, vy=%.2f, ax=%.2f, ay=%.2f' % (self.ship_1.vx,self.ship_1.vy, self.ship_1.ax, self.ship_1.ay), False, (255, 255, 255))
            self.game.window.blit(ship_va, (DEBUG_TEXT_XPOS + 5, 55))

            ship_angle = self.myfont.render('Angle: %s' % (self.ship_1.angle,), False, (255, 255, 255))
            self.game.window.blit(ship_angle, (DEBUG_TEXT_XPOS + 5, 80))

            dt = self.myfont.render('Frames: %s' % (self.frames,), False, (255, 255, 255))
            self.game.window.blit(dt, (DEBUG_TEXT_XPOS + 5, 105))

            fps = self.myfont.render('FPS: %.2f' % self.clock.get_fps(), False, (255, 255, 255))
            self.game.window.blit(fps, (DEBUG_TEXT_XPOS + 5, 130))

            #ship_lives = self.myfont.render('Lives: %s' % (self.ship_1.lives,), False, (255, 255, 255))
            #self.game.window.blit(ship_lives, (DEBUG_TEXT_XPOS + 5, 105))

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

    parser.add_argument('-width', '--width', help='', type=int, action="store", default=1920)
    parser.add_argument('-height', '--height', help='', type=int, action="store", default=1080)
    parser.add_argument('-np', '--nb_player', help='', type=int, action="store", default=2)

    parser.add_argument('-m', '--motion', help='How the ship moves', action="store", default='gravity', choices=("basic", "thrust", "gravity"))
    parser.add_argument('-r', '--record_play', help='', action="store", default="")
    parser.add_argument('-pr', '--play_recorded', help='', action="store", default="")
    parser.add_argument('-s', '--sensor', help='', action="store", default="", choices=("octo", "octo_fixed", "beam", ""))
    parser.add_argument('-rm', '--run_mode', help='', action="store", default="game", choices=("game", "training", ))

    result = parser.parse_args()
    args = dict(result._get_kwargs())

    print("Args", args)

    # env
    seq = Sequence(args["width"], args["height"], args["nb_player"], mode=args["run_mode"], motion=args["motion"], sensor=args["sensor"], record_play=args["record_play"], play_recorded=args["play_recorded"])
    seq.main_loop()

if __name__ == '__main__':
    run()
