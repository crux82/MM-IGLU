
import inflect
import random
import string


# mapping the block colors
block_color_map = {
    #0: "air",
    57: "blue",
    50: "yellow",
    59: "green",
    47: "orange",
    56: "purple",
    60: "red"
}

# mapping the block colors
block_color_map_alternative = {
    -1: "white",
    0: "grey",
    1: "blue",
    2: "green",
    3: "red",
    4: "orange",
    5: "purple",
    6: "yellow",
}

alphabet = string.ascii_lowercase + string.digits

class Block:
    def __init__(self, x, y, z, color_number) -> None:
        self.id = ''.join(random.choices(alphabet, k=8))
        self.x = x
        self.y = y
        self.z = z
        try:
            self.color = block_color_map[color_number]
        except KeyError:
            self.color = block_color_map_alternative[color_number]
        except Exception as e:
            print(e)
            print("BLOCK CLASS")
            
    
    def __str__(self) -> None:
        return f"ID: {self.id} \tcolor: {self.color} \t(x: {self.x} \ty {self.y} \tz: {self.z})"


def textify_environment(gridworld_state):
    p = inflect.engine()
    colours = set()
    colours_dict = {}
    map_description = "there are "

    raw_blocks = gridworld_state['worldEndingState']['blocks']
    for x_b,y_b,z_b,color_n in raw_blocks:
        b = Block(x_b,y_b,z_b,color_n)
        
        if b.color not in colours:
            colours.add(b.color)
            colours_dict[b.color] = {
                "count": 1,
                'on_ground': 1 if b.x == -5 else 0
            }
        else:
            colours_dict[b.color]['count'] += 1
            colours_dict[b.color]['on_ground'] += 1 if b.x == -5 else 0
            

    for b_col in block_color_map:
        if block_color_map[b_col] not in colours_dict:
            map_description += f"no {block_color_map[b_col]} blocks, "
        else:
            map_description += f"{p.number_to_words(colours_dict[block_color_map[b_col]]['count'])} {block_color_map[b_col]} blocks, "
            if colours_dict[block_color_map[b_col]]['on_ground'] > 0:
                map_description += f"{p.number_to_words(colours_dict[block_color_map[b_col]]['on_ground'])} of which are on the ground, "
    map_description = map_description[:len(map_description)-2]

    return map_description