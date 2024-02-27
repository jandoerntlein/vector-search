from angle_emb import AnglE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Simple profiling function from https://stackoverflow.com/a/11151365
import time
def st_time(func):
    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r
    return st_func

class Database:
    angle       = None

    # Poor man's data structure. A compound data type would be much better here, to ensure data integrity. 
    data_text   = []
    data_vector = []

    @st_time
    def __init__(self):
        # .cuda() is also used for MPS devices (Mac), you might also want to try without this conversion if you face errors.
        self.angle = AnglE.from_pretrained('./UAE-Large-V1', pooling_strategy='cls').cuda() 
    
    def __embedding__(self, text:str):
        return self.angle.encode(text, to_numpy=True)

    @st_time
    def add(self, text:str):
        self.data_text.append(text)
        self.data_vector.append(self.__embedding__(text))
        assert len(self.data_text) == len(self.data_vector)

    def print(self):
        for i in range(len(self.data_text)):
            print(self.data_text[i] + " -> " + str(self.data_vector[i]))

    @st_time
    def search(self, query:str, limit:int):
        search_vector  = self.__embedding__(query)
        cos_similarity = [cosine_similarity(search_vector, v) for v in self.data_vector]

        # sort, flatten, reverse the index vector, which now points to the results
        indices = np.argsort(cos_similarity, axis=0).flatten()[::-1] 

        # return an iterable, with indices pointing to data_text entries
        for idx in indices[:limit]:
            yield self.data_text[idx]

# Create a sample database, and fill it with test data
db = Database()

print("--- DATABASE CREATION ---")
db.add("Apple iPhone 13 - A smartphone with an A15 Bionic chip and advanced dual-camera system.")
db.add("Samsung Galaxy S21 - Android smartphone with 5G connectivity and high-resolution cameras.")
db.add("Sony WH-1000XM4 - Wireless noise-canceling headphones with long battery life.")
db.add("Instant Pot DUO60 - 6-quart multi-cooker that can pressure cook, slow cook, and more.")
db.add("LEGO Classic Medium Creative Brick Box - Set of colorful LEGO bricks for creative building.")
db.add("Nintendo Switch - Hybrid gaming console for handheld and TV-connected play.")
db.add("Fitbit Charge 4 - Fitness tracker with heart rate monitoring and built-in GPS.")
db.add("Dyson V11 Torque Drive - Cordless vacuum cleaner with intelligent suction adjustment.")
db.add("Keurig K-Classic Coffee Maker - Single-serve coffee maker with a large water reservoir.")
db.add("The North Face Borealis Backpack - Versatile backpack with a protective laptop compartment.")
db.add("Bose QuietComfort 35 II - Wireless Bluetooth headphones with noise cancellation.")
db.add("Amazon Echo Dot (4th Gen) - Smart speaker with Alexa for voice control.")
db.add("GoPro HERO9 Black - Action camera with 5K video and 20MP photo capability.")
db.add("Kindle Paperwhite - Waterproof e-reader with a high-resolution display.")
db.add("Anker PowerCore 10000 - Compact portable charger with high-speed charging.")
db.add("Philips Hue White and Color Ambiance Starter Kit - Smart LED lighting system with color control.")
db.add("Adidas Ultraboost Running Shoes - Comfortable running shoes with responsive cushioning.")
db.add("Roku Streaming Stick+ - Portable streaming device with 4K and HDR support.")
db.add("Logitech MX Master 3 - Advanced wireless mouse with customizable buttons.")
db.add("Cuisinart Air Fryer Toaster Oven - Versatile countertop appliance for healthier cooking.")
db.add("Ring Video Doorbell 3 - Wi-Fi-enabled smart doorbell with HD video and motion detection.")
db.add("YETI Rambler 20 oz Tumbler - Stainless steel insulated tumbler for hot and cold beverages.")
db.add("Canon EOS Rebel T7 DSLR Camera - Entry-level DSLR with 24.1 MP and built-in Wi-Fi.")
db.add("Peloton Bike - Indoor exercise bike with live and on-demand fitness classes.")
db.add("Patagonia Nano Puff Jacket - Lightweight and packable insulated jacket for outdoor activities.")
db.add("Microsoft Surface Pro 7 - 2-in-1 laptop with a detachable touchscreen.")
db.add("Oculus Quest 2 - All-in-one virtual reality headset for immersive gaming.")
db.add("Gillette Fusion5 ProGlide Razor - Men's razor with FlexBall technology for a close shave.")
db.add("KitchenAid Artisan Stand Mixer - Durable mixer for baking and cooking with various attachments.")
db.add("Brita Ultra Max Filtering Dispenser - Large capacity water filtration system for clean drinking water.")
db.add("Sony PlayStation 5 - Next-generation gaming console with powerful performance.")
db.add("Revlon One-Step Hair Dryer and Volumizer - Hot air brush for easy hair styling.")
db.add("Trek Domane SL 6 Road Bike - Performance road bike with a comfortable and lightweight design.")
db.add("Weber Original Kettle Premium Charcoal Grill - Classic charcoal grill with a one-touch cleaning system.")
db.add("JBL Flip 5 - Portable Bluetooth speaker with waterproof design and powerful sound.")
db.add("Garmin Forerunner 245 - GPS running watch with advanced training features.")
db.add("L.L.Bean Women's Bean Boots - Iconic duck boots designed for wet weather and outdoor activities.")
db.add("Moleskine Classic Notebook - Durable notebook for writing, sketching, and journaling.")
db.add("DeWalt 20V MAX Cordless Drill Combo Kit - High-performance drill and impact driver set.")
db.add("Polaroid Originals OneStep+ - Instant camera with Bluetooth connectivity for creative photography.")
db.add("Vitamix 5200 Blender - Professional-grade blender for smoothies, soups, and more.")
db.add("UGG Women's Classic Short II Boot - Cozy sheepskin boots for warmth and comfort.")
db.add("Asus ROG Strix GeForce RTX 3080 - High-end gaming graphics card with ray tracing technology.")
db.add("Champion Men's Powerblend Fleece Hoodie - Soft and comfortable hoodie for casual wear.")
db.add("Theragun PRO - Percussive therapy device for deep muscle treatment.")
db.add("Allbirds Wool Runners - Sustainable and comfortable shoes made from merino wool.")
db.add("Fujifilm X-T4 Mirrorless Camera - High-performance camera with in-body image stabilization.")
db.add("NETGEAR Nighthawk AX12 - Wi-Fi 6 router for faster and more reliable internet connectivity.")
db.add("Celestron NexStar 8SE Telescope - Computerized telescope for viewing planets and deep-sky objects.")
db.add("Breville Bambino Plus Espresso Machine - Compact espresso machine with automatic milk frothing.")
db.add("Black Diamond Spot 350 Headlamp - Bright and durable headlamp for outdoor adventures.")

# print the whole database
print("--- DATABASE CONTENT ---")
db.print()

# run some example queries with a quick helper function
print("--- DATABASE QUERY ---")
def simulate_search(query):
    results = db.search(query, 3) # limit to the top 3 results
    print("Since you are looking for <" + query + ">, you also might want to have a look at: ")
    for e in results:
        print("  - " + e)

simulate_search("shirt")
simulate_search("gaming gpu")
simulate_search("coffee machine")
