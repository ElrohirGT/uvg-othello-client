import requests
import sys
import time
from othello_ai import ai_move

BASE_URL = "http://localhost:8000"
# BASE_URL = 'https://7b679617-8c6b-4d0f-bb51-0505412c6c17.us-east-1.cloud.genez.io'

if __name__ == "__main__":


    if len(sys.argv) != 3:
        print("Usage: python othello_player.py <Session Name> <Username>")
        sys.exit(1)

    tournament_name = sys.argv[1]
    username = sys.argv[2]

    print('Requesting to join!')
    req = requests.post(f"{BASE_URL}/tournament/join", json = {
        'username' : username 
        , 'tournament_name' : tournament_name
    })

    print(req)

    if req.status_code == 409: 
        print(req.json()['detail'])

    if req.status_code == 200: 
        print(f'Welcome to the {tournament_name} tournament. Please await while the tournament starts.')

    
        while True: 

            active = requests.post(f"{BASE_URL}/match/active", json = {
                'username' : username 
                , 'tournament_name' : tournament_name
            })
            
            if active.json()['is_in_active_match']: 

                while True: 
                    status = requests.post(f"{BASE_URL}/match/status", json = {
                        'username' : username 
                        , 'tournament_name' : tournament_name
                    })
                    
                    if status.status_code == 404:
                        break
                    if status.status_code == 409: #Is not your turn 
                        time.sleep(2)
                    if status.status_code == 200: #Is your turn 

                        response = status.json() 
                        if response['msg'] == 'Match ended':
                            print(f"Winner is {response['winner']}!")
                        else:
                            moved = False 

                            while not moved: 
                                
                                board = status.json()['board'] 
                                player = status.json()['player_color']
                                move = ai_move(board, player)
                                print(f'Your move is {move}')
                                if move is None:
                                    moved = True
                                else:
                                    res = requests.post(f"{BASE_URL}/match/move", json={
                                        "username" : username
                                        , "tournament_name": tournament_name
                                        , "x" : move[0]
                                        , "y": move[1]
                                    })
                                    
                                    if res.status_code == 409: 
                                        print('Invalid movement!!!')
                                    else: 
                                        moved = True
                            
            else:
                print('Await for your next match')
                time.sleep(10)
