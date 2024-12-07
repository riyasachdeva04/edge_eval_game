import game1
import game2
import temp_state
import temp_state2

if __name__ == "__main__":
    selected_level = game1.run_game1() 
    temp_state.run_game()
    if selected_level is not None: 
        game2.run_game(selected_level)  
    else:
        print("Game1 encountered an issue.")
    temp_state2.run_game()
