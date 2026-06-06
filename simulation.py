import pygame
import sys
import random
import math
import numpy as np
import cv2

DAY_NUMBER = 6

class Shelter:
    """Ajanların saklanabileceği sığınak. İçinde sınırlı süre kalınabilir."""
    def __init__(self, x, y, radius=40):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius
        self.max_occupancy = 3  # Aynı anda en fazla 3 ajan sığabilir
        self.current_occupants = []  # Şu an içeride olan ajanlar

    def is_inside(self, agent):
        """Ajanın merkezi sığınağın içinde mi kontrol et."""
        cx = agent.x + agent.size / 2
        cy = agent.y + agent.size / 2
        dist = math.hypot(cx - self.x, cy - self.y)
        return dist <= self.radius

    def can_enter(self, agent):
        """Ajan sığınağa girebilir mi?"""
        if agent.shelter_cooldown > 0:
            return False
        if len(self.current_occupants) >= self.max_occupancy:
            return False
        return True

    def draw(self, surface):
        # Yarı-saydam sığınak çemberi
        shelter_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        # İç dolgu - yarı saydam
        pygame.draw.circle(shelter_surface, (100, 200, 200, 40), (self.radius, self.radius), self.radius)
        # Dış çizgi
        pygame.draw.circle(shelter_surface, (100, 200, 200, 120), (self.radius, self.radius), self.radius, 3)
        surface.blit(shelter_surface, (int(self.x - self.radius), int(self.y - self.radius)))

        # Doluluk göstergesi
        occ_text_color = (100, 200, 200)
        if len(self.current_occupants) >= self.max_occupancy:
            occ_text_color = (255, 80, 80)
        font = pygame.font.SysFont("Arial", 20, bold=True)
        occ_label = font.render(f"{len(self.current_occupants)}/{self.max_occupancy}", True, occ_text_color)
        surface.blit(occ_label, (int(self.x - occ_label.get_width() / 2), int(self.y + self.radius + 4)))


class Agent:
    def __init__(self, x, y, team=0, size=15, brain=None):
        self.x = float(x)
        self.y = float(y)
        self.team = team
        self.size = size
        self.fitness = 0  
        self.speed = 5.0
        
        # TAKIMLARA ÖZEL STATLAR (Mavi: Tank/Melee, Sarı: Range)
        if self.team == 0:
            self.color = (0, 150, 255)
            self.max_health = 200        # TANK: 2 kat can
            self.damage_multiplier = 0.5 # %50 daha az hasar alır
            self.combat_range = 15      # MELEE: Sadece dibine girince vurabilir
        else:
            self.color = (255, 255, 0)
            self.max_health = 100        # Normal can
            self.damage_multiplier = 1.0 # Normal hasar alır
            self.combat_range = 45      # RANGE: Uzaktan vurabilir
        
        self.health = self.max_health
        
        # Sığınak durumu
        self.in_shelter = False          # Şu an sığınakta mı?
        self.shelter_timer = 0           # Sığınakta ne kadar kaldı (frame)
        self.shelter_cooldown = 0        # Sığınaktan çıktıktan sonra bekleme süresi
        self.max_shelter_time = 180      # Sığınakta kalınabilecek max süre (~3 sn @ 60fps)
        self.shelter_cooldown_time = 180 # Çıktıktan sonra tekrar giremez (~3 sn)
        self.current_shelter = None      # Hangi sığınakta
        
        if brain is not None:
            self.brain = brain.copy()
        else:
            # Önce beyni çok hafif rastgelelikte başlatıyoruz (Evrim için çeşitlilik lazım)
            self.brain = np.random.randn(12, 2) * 0.1 
            
            # --- İÇGÜDÜLER (HEURISTICS) EKLENİYOR ---
            
            # 1. Yemeğe Gitme İçgüdüsü (Giriş 0 ve 1 -> Çıkış 0 ve 1)
            # Yemeğin X yönü, ajanın X hızını (vx) pozitif etkilesin
            self.brain[0, 0] += 2.0 
            # Yemeğin Y yönü, ajanın Y hızını (vy) pozitif etkilesin
            self.brain[1, 1] += 2.0 
            
            # 2. Avcıdan Kaçma İçgüdüsü (Giriş 3 ve 4 -> Çıkış 0 ve 1)
            # Avcının X yönü, ajanın X hızını (vx) NEGATİF etkilesin (Tersine koşsun)
            self.brain[3, 0] -= 2.5 
            # Avcının Y yönü, ajanın Y hızını (vy) NEGATİF etkilesin
            self.brain[4, 1] -= 2.5 
            
            # 3. Sığınağa Gitme İçgüdüsü (Giriş 9 ve 10 -> Çıkış 0 ve 1)
            # Hafif bir sığınak eğilimi veriyoruz
            self.brain[9, 0] += 1.0
            self.brain[10, 1] += 1.0
            
        self.vx = 0.0
        self.vy = 0.0

    # Görüş mesafesi kısıtlaması kaldırıldı (Sınırsız)
    def get_nearest_hunter(self, hunters):
        if not hunters:
            return 0.0, 0.0
            
        min_dist = float('inf')
        nearest_hunter = None
        agent_center_x = self.x + self.size / 2
        agent_center_y = self.y + self.size / 2
        
        for hunter in hunters:
            dist = math.hypot(hunter.x - agent_center_x, hunter.y - agent_center_y)
            if dist < min_dist:
                min_dist = dist
                nearest_hunter = hunter
                
        if nearest_hunter:
            dx = nearest_hunter.x - agent_center_x
            dy = nearest_hunter.y - agent_center_y
            return dx, dy
        return 0.0, 0.0

    def get_nearest_enemy(self, alive_agents):
        if not alive_agents:
            return 0.0, 0.0
            
        min_dist = float('inf')
        nearest_enemy = None
        agent_center_x = self.x + self.size / 2
        agent_center_y = self.y + self.size / 2
        
        for other in alive_agents:
            if other.team != self.team:
                dist = math.hypot(other.x - agent_center_x, other.y - agent_center_y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_enemy = other
                    
        if nearest_enemy:
            dx = nearest_enemy.x - agent_center_x
            dy = nearest_enemy.y - agent_center_y
            return dx, dy
        return 0.0, 0.0

    def get_nearest_food(self, foods):
        if not foods:
            return 0.0, 0.0
            
        min_dist = float('inf')
        nearest_food = None
        agent_center_x = self.x + self.size / 2
        agent_center_y = self.y + self.size / 2
        
        for food in foods:
            dist = math.hypot(food.x - agent_center_x, food.y - agent_center_y)
            if dist < min_dist:
                min_dist = dist
                nearest_food = food
                
        if nearest_food:
            dx = nearest_food.x - agent_center_x
            dy = nearest_food.y - agent_center_y
            return dx, dy
        return 0.0, 0.0

    def get_nearest_shelter(self, shelters):
        """En yakın sığınağın yönünü ve mesafesini döndürür."""
        if not shelters:
            return 0.0, 0.0
            
        min_dist = float('inf')
        nearest_shelter = None
        agent_center_x = self.x + self.size / 2
        agent_center_y = self.y + self.size / 2
        
        for shelter in shelters:
            dist = math.hypot(shelter.x - agent_center_x, shelter.y - agent_center_y)
            if dist < min_dist:
                min_dist = dist
                nearest_shelter = shelter
                
        if nearest_shelter:
            dx = nearest_shelter.x - agent_center_x
            dy = nearest_shelter.y - agent_center_y
            return dx, dy
        return 0.0, 0.0

    def enter_shelter(self, shelter):
        """Sığınağa gir."""
        self.in_shelter = True
        self.shelter_timer = 0
        self.current_shelter = shelter
        shelter.current_occupants.append(self)

    def leave_shelter(self):
        """Sığınaktan çık."""
        if self.current_shelter and self in self.current_shelter.current_occupants:
            self.current_shelter.current_occupants.remove(self)
        self.in_shelter = False
        self.shelter_timer = 0
        self.shelter_cooldown = self.shelter_cooldown_time
        self.current_shelter = None

    def update(self, width, height, foods, hunters, alive_agents, shelters=None):
        if shelters is None:
            shelters = []
            
        # Cooldown sayacını azalt
        if self.shelter_cooldown > 0:
            self.shelter_cooldown -= 1
        
        # Sığınaktayken: timer artar, can kaybı YOK
        if self.in_shelter:
            self.shelter_timer += 1
            
            # Süre doldu! Zorla çık
            if self.shelter_timer >= self.max_shelter_time:
                self.leave_shelter()
            else:
                # Sığınaktayken hareket etme
                self.vx = 0
                self.vy = 0
                return
        else:
            self.health -= 1 
        
        # Maksimum çapraz uzaklık (Ekranın köşegen uzunluğu)
        max_dist = math.hypot(width, height)

        food_dx, food_dy = self.get_nearest_food(foods)
        dist_food = math.hypot(food_dx, food_dy)
        if dist_food > 0:
            food_dx /= dist_food
            food_dy /= dist_food

        hunter_dx, hunter_dy = self.get_nearest_hunter(hunters)
        dist_hunter = math.hypot(hunter_dx, hunter_dy)
        if dist_hunter > 0:
            hunter_dx /= dist_hunter
            hunter_dy /= dist_hunter
            
        enemy_dx, enemy_dy = self.get_nearest_enemy(alive_agents)
        dist_enemy = math.hypot(enemy_dx, enemy_dy)
        if dist_enemy > 0:
            enemy_dx /= dist_enemy
            enemy_dy /= dist_enemy

        shelter_dx, shelter_dy = self.get_nearest_shelter(shelters)
        dist_shelter = math.hypot(shelter_dx, shelter_dy)
        if dist_shelter > 0:
            shelter_dx /= dist_shelter
            shelter_dy /= dist_shelter
            
        # 12 giriş: food(3) + hunter(3) + enemy(3) + shelter(3)
        inputs = np.array([
            food_dx, food_dy, dist_food / max_dist,
            hunter_dx, hunter_dy, dist_hunter / max_dist,
            enemy_dx, enemy_dy, dist_enemy / max_dist,
            shelter_dx, shelter_dy, dist_shelter / max_dist
        ])
        
        outputs = np.dot(inputs, self.brain)
        
        self.vx = outputs[0]
        self.vy = outputs[1]
        
        current_speed = math.hypot(self.vx, self.vy)
        if current_speed > 0:
            self.vx = (self.vx / current_speed) * self.speed
            self.vy = (self.vy / current_speed) * self.speed

        self.x += self.vx
        self.y += self.vy

        if self.x <= 0:
            self.x = 0
            self.vx *= -1
        elif self.x >= width - self.size:
            self.x = width - self.size
            self.vx *= -1
            
        if self.y <= 0:
            self.y = 0
            self.vy *= -1
        elif self.y >= height - self.size:
            self.y = height - self.size
            self.vy *= -1

    def draw(self, surface):
        # Sığınaktaysa yarı saydam çiz
        if self.in_shelter:
            agent_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            r, g, b = self.color
            pygame.draw.rect(agent_surface, (r, g, b, 100), (0, 0, self.size, self.size))
            surface.blit(agent_surface, (int(self.x), int(self.y)))
            
            # Sığınak süresi barı (turuncu)
            bar_width = self.size
            bar_height = 3
            shelter_ratio = 1.0 - (self.shelter_timer / self.max_shelter_time)
            pygame.draw.rect(surface, (80, 80, 80), (int(self.x), int(self.y - 16), bar_width, bar_height))
            pygame.draw.rect(surface, (255, 165, 0), (int(self.x), int(self.y - 16), int(bar_width * shelter_ratio), bar_height))
        else:
            # Ajanın kendisi
            rect = pygame.Rect(int(self.x), int(self.y), self.size, self.size)
            pygame.draw.rect(surface, self.color, rect)
        
        # Can Barı Çizimi
        bar_width = self.size
        bar_height = 4
        health_ratio = max(0, self.health / self.max_health)
        pygame.draw.rect(surface, (255, 0, 0), (int(self.x), int(self.y - 10), bar_width, bar_height))
        pygame.draw.rect(surface, (0, 255, 0), (int(self.x), int(self.y - 10), int(bar_width * health_ratio), bar_height))
        
        # Savaş Menzilini (Combat Range) çember olarak çiziyoruz
        center_x = int(self.x + self.size / 2)
        center_y = int(self.y + self.size / 2)
        pygame.draw.circle(surface, self.color, (center_x, center_y), self.combat_range, 1)
        
        # Cooldown göstergesi (cooldown varken küçük mavi çember)
        if self.shelter_cooldown > 0:
            cooldown_ratio = self.shelter_cooldown / self.shelter_cooldown_time
            pygame.draw.arc(surface, (100, 200, 200), 
                          (int(self.x - 2), int(self.y - 2), self.size + 4, self.size + 4),
                          0, cooldown_ratio * 2 * math.pi, 2)
        
    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.size, self.size)


class Food:
    def __init__(self, x, y, radius=8):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 255, 0), (int(self.x), int(self.y)), self.radius)

    def respawn(self, width, height):
        self.x = float(random.randint(self.radius, width - self.radius))
        self.y = float(random.randint(self.radius, height - self.radius))
        
    def get_rect(self):
        return pygame.Rect(int(self.x - self.radius), int(self.y - self.radius), self.radius * 2, self.radius * 2)


class Hunter:
    def __init__(self, x, y, radius=10):
        self.x = float(x)
        self.y = float(y)
        self.radius = radius
        self.speed = 3.5
        self.health = 500 
        self.max_health = 1000  # CAN SINIRI: Avcılar çok fazla güçlenemez
    def update(self, agents):
        self.health -= 1
        
        # Sığınaktaki ajanları GÖRME - sadece dışarıdakileri hedefle
        visible_agents = [a for a in agents if not a.in_shelter]
        
        if not visible_agents:
            return
            
        min_dist = float('inf')
        nearest_agent = None
        
        for agent in visible_agents:
            dist = math.hypot(agent.x + agent.size/2 - self.x, agent.y + agent.size/2 - self.y)
            if dist < min_dist:
                min_dist = dist
                nearest_agent = agent
                
        if nearest_agent:
            dx = (nearest_agent.x + nearest_agent.size/2) - self.x
            dy = (nearest_agent.y + nearest_agent.size/2) - self.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                vx = (dx / dist) * self.speed
                vy = (dy / dist) * self.speed
                
                self.x += vx
                self.y += vy

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 0, 0), (int(self.x), int(self.y)), self.radius)
        
        # Avcı Can Barı
        bar_width = self.radius * 2.5
        bar_height = 5
        health_ratio = max(0, self.health / self.max_health)
        pygame.draw.rect(surface, (100, 0, 0), (int(self.x - bar_width/2), int(self.y - self.radius - 12), bar_width, bar_height))
        pygame.draw.rect(surface, (255, 50, 50), (int(self.x - bar_width/2), int(self.y - self.radius - 12), int(bar_width * health_ratio), bar_height))

    def get_rect(self):
        return pygame.Rect(int(self.x - self.radius), int(self.y - self.radius), self.radius * 2, self.radius * 2)


class Simulation:
    def __init__(self):
        pygame.init()
        
        self.sim_width = 720
        self.sim_height = 1280
        self.display_surface = pygame.Surface((self.sim_width, self.sim_height))
        
        self.win_width = 360
        self.win_height = 640
        self.actual_screen = pygame.display.set_mode((self.win_width, self.win_height))
        
        pygame.display.set_caption("AI Evolution Habitat - Shorts Edition")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 45, bold=True)
        
        self.generation = 1
        
        self.agents = []
        self.dead_agents = []
        self.foods = []
        self.hunters = []
        self.shelters = []
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_filename = f'day_{DAY_NUMBER}_tribalism.mp4'
        self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, 60.0, (self.sim_width, self.sim_height))
        
        if not self.video_writer.isOpened():
            print(f"HATA: Video yazıcı başlatılamadı! codec: mp4v, dosya: {self.output_filename}")
        
        self.init_generation(initial=True)

    def init_generation(self, initial=False):
        self.agents = []
        self.dead_agents = []
        self.hunters = []
        
        for _ in range(3):
            hx = random.randint(10, self.sim_width - 10)
            hy = random.randint(10, self.sim_height - 10)
            self.hunters.append(Hunter(hx, hy))
        
        self.foods = []
        for _ in range(80):
            fx = random.randint(8, self.sim_width - 8)
            fy = random.randint(8, self.sim_height - 8)
            self.foods.append(Food(fx, fy))
        
        # Sığınaklar - haritanın farklı bölgelerine yerleştir
        self.shelters = []
        shelter_positions = [
            (self.sim_width * 0.25, self.sim_height * 0.25),
            (self.sim_width * 0.75, self.sim_height * 0.25),
            (self.sim_width * 0.25, self.sim_height * 0.75),
            (self.sim_width * 0.75, self.sim_height * 0.75),
        ]
        for sx, sy in shelter_positions:
            self.shelters.append(Shelter(sx, sy))
        
        if initial:
            for i in range(50):
                team = 0 if i < 25 else 1
                ax = random.randint(0, self.sim_width - 15)
                ay = random.randint(0, self.sim_height - 15)
                self.agents.append(Agent(ax, ay, team=team))

    def reset_generation(self):
        all_agents = self.agents + self.dead_agents
        if not all_agents:
            self.init_generation(initial=True)
            return

        all_agents.sort(key=lambda a: a.fitness, reverse=True)
        champions = all_agents[:5]
        
        self.generation += 1
        
        self.agents = []
        self.dead_agents = []
        self.hunters = []
        
        for _ in range(3):
            hx = random.randint(10, self.sim_width - 10)
            hy = random.randint(10, self.sim_height - 10)
            self.hunters.append(Hunter(hx, hy))
        
        self.foods = []
        for _ in range(80):
            fx = random.randint(8, self.sim_width - 8)
            fy = random.randint(8, self.sim_height - 8)
            self.foods.append(Food(fx, fy))
        
        # Sığınakları yeniden yerleştir
        self.shelters = []
        shelter_positions = [
            (self.sim_width * 0.25, self.sim_height * 0.25),
            (self.sim_width * 0.75, self.sim_height * 0.25),
            (self.sim_width * 0.25, self.sim_height * 0.75),
            (self.sim_width * 0.75, self.sim_height * 0.75),
        ]
        for sx, sy in shelter_positions:
            self.shelters.append(Shelter(sx, sy))
        
        for i in range(50):
            team = 0 if i < 25 else 1
            parent = random.choice(champions)
            new_brain = parent.brain.copy()
            
            mutation_mask = np.random.randn(*new_brain.shape) * 0.15
            new_brain += mutation_mask
            
            ax = random.randint(0, self.sim_width - 15)
            ay = random.randint(0, self.sim_height - 15)
            new_agent = Agent(ax, ay, team=team, brain=new_brain)
            self.agents.append(new_agent)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                if event.key == pygame.K_n:
                    self.reset_generation()
        return True

    def update(self):
        if not self.agents:
            self.reset_generation()
            return
            
        alive_hunters = []
        for hunter in self.hunters:
            hunter.update(self.agents)
            if hunter.health > 0:
                alive_hunters.append(hunter)
        self.hunters = alive_hunters

        handled_collisions = set()
        
        for agent in self.agents:
            # Beyni güncelle ve hareket et
            agent.update(self.sim_width, self.sim_height, self.foods, self.hunters, self.agents, self.shelters)
            
            # Sığınaktaysa hasar almaz, savaşamaz, yemek yiyemez
            if agent.in_shelter:
                continue
            
            # Sığınağa giriş kontrolü
            for shelter in self.shelters:
                if shelter.is_inside(agent) and shelter.can_enter(agent) and not agent.in_shelter:
                    agent.enter_shelter(shelter)
                    break
            
            if agent.in_shelter:
                continue
            
            agent_rect = agent.get_rect()
            
            # Avcı Etkileşimleri (Hasar ve Temas)
            for hunter in self.hunters:
                hunter_center_x, hunter_center_y = hunter.x, hunter.y
                agent_center_x, agent_center_y = agent.x + agent.size / 2, agent.y + agent.size / 2
                dist = math.hypot(agent_center_x - hunter_center_x, agent_center_y - hunter_center_y)
                
                # Menzilli Saldırı: Ajan avcıya vurur
                if dist <= agent.combat_range + hunter.radius:
                    hunter.health -= 2 # Avcılar ajanlardan saniyede ciddi hasar yer
                
                # Fiziksel Temas: Avcı ajanı yer
                hunter_rect = hunter.get_rect()
                if agent_rect.colliderect(hunter_rect):
                    agent.health -= 150 * agent.damage_multiplier
                    # Can kazanırken üst sınırı geçemez
                    hunter.health = min(hunter.health + 100, hunter.max_health) 
                    break
                    
            # Ajan-Ajan Çarpışmaları ve Menzilli Vuruşlar
            if agent.health > 0:
                for other in self.agents:
                    if agent != other and other.health > 0 and not other.in_shelter:
                        pair = frozenset([agent, other])
                        if pair not in handled_collisions:
                            # İki ajanın merkezleri arasındaki mesafe
                            cx1 = agent.x + agent.size / 2
                            cy1 = agent.y + agent.size / 2
                            cx2 = other.x + other.size / 2
                            cy2 = other.y + other.size / 2
                            dist = math.hypot(cx1 - cx2, cy1 - cy2)
                            
                            if agent.team != other.team:
                                hit_occurred = False
                                
                                # Eğer 'other' ajanı 'agent'in menziline girdiyse hasar ver
                                if dist <= agent.combat_range:
                                    other.health -= 15 * other.damage_multiplier
                                    hit_occurred = True
                                    
                                # Eğer 'agent', 'other' ajanının menziline girdiyse hasar al
                                if dist <= other.combat_range:
                                    agent.health -= 15 * agent.damage_multiplier
                                    hit_occurred = True
                                    
                                # Eğer bir menzilli saldırı veya çarpışma olduysa sekmelerini sağla
                                if hit_occurred:
                                    handled_collisions.add(pair)
                                    agent.vx *= -1
                                    agent.vy *= -1
                                    other.vx *= -1
                                    other.vy *= -1
                            else:
                                # Aynı takımdakiler sadece fiziksel olarak dip dibe (15 px) girerse seker, hasar almaz
                                if dist <= 15:
                                    handled_collisions.add(pair)
                                    agent.vx *= -1
                                    agent.vy *= -1
                                    other.vx *= -1
                                    other.vy *= -1
            
            # Yemek yeme (Fiziksel temas gerektirir)
            if agent.health > 0:
                for food in self.foods:
                    food_rect = food.get_rect()
                    if agent_rect.colliderect(food_rect):
                        agent.health = min(agent.health + 25, agent.max_health) 
                        agent.fitness += 1
                        food.respawn(self.sim_width, self.sim_height)
            
        alive_agents = []
        for agent in self.agents:
            if agent.health > 0:
                alive_agents.append(agent)
            else:
                # Ölürken sığınaktaysa çıkar
                if agent.in_shelter:
                    agent.leave_shelter()
                self.dead_agents.append(agent)
                
        self.agents = alive_agents

    def draw(self):
        self.display_surface.fill((0, 0, 0))
        
        # Önce sığınakları çiz (arka plan)
        for shelter in self.shelters:
            shelter.draw(self.display_surface)
            
        for food in self.foods:
            food.draw(self.display_surface)
            
        for agent in self.agents:
            agent.draw(self.display_surface)
            
        for hunter in self.hunters:
            hunter.draw(self.display_surface)
            
        day_text = self.font.render(f"DAY: {DAY_NUMBER}", True, (255, 255, 255))
        gen_text = self.font.render(f"GEN: {self.generation}", True, (255, 255, 255))
        
        current_best_fit = max([a.fitness for a in self.agents]) if self.agents else 0
        fit_text = self.font.render(f"BEST FIT: {current_best_fit}", True, (200, 200, 255))
        
        # Sığınakta olan ajan sayısı

        self.display_surface.blit(day_text, (20, 50))
        self.display_surface.blit(gen_text, (20, 100))
        self.display_surface.blit(fit_text, (20, 150))
     
        
        scaled_surface = pygame.transform.scale(self.display_surface, (self.win_width, self.win_height))
        self.actual_screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            
            frame = pygame.surfarray.array3d(self.display_surface)
            frame = np.transpose(frame, (1, 0, 2))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
            
            self.clock.tick(60)
            
        self.video_writer.release()
        print(f"Video başarıyla kaydedildi: {self.output_filename}")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()