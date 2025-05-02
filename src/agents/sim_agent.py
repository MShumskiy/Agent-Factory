from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os

import requests
from typing import List, Dict, Optional
from src.utils.llmp_utils import llmp_call
import numpy as np

class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: Optional[dict] = None
    image: Optional[str] = None
    tools: Optional[List[Dict]] = None
    src: str = None
    temperature: float = 0.5
    
class SIMAgent:
    
    def __init__(self, model, kb_agent,adv_agent):
        
        
        from dotenv import load_dotenv
        from sentence_transformers import CrossEncoder
        
        load_dotenv()
        
        self.src = 'sim_agent'
        self.model = model
        self.llmp_url = os.getenv("LLMP_URL")
        self.llmp_password = os.getenv("LLMP_PASSWORD")
        #self.tools_desc = tools_desc
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.kb_agent = kb_agent
        self.adv_agent = adv_agent

                
    def judge(self,moves):
        
        play = ''
        for move in moves:
            #print(f"\n**{move}**:\n {moves[move]}\n\n**CHANGE PLAYER**\n")
            play = play + f"\n**{move}**:\n {moves[move]}\n\n**CHANGE PLAYER**\n"
            
        judge_system_prompt = 'You are a judge in a turn based game. You are given the moves of both players. Yu must analyze all their moves and determine the end result. You are not on any side, you are unbiased and just provide the end status of the game. You need to determine which player has the advantage based on the moves they made. Provide your reasoning and the final decision. There are 2 players, adv_1 and adv_2. The moves are as follows:\n\n'    
        judge_prompt = play + """\n\n Evaluate the game in a very objective manner.
        Provide the following: Game Summary, Player 1 Stauts, Player 2 Status, Outcome So Far, Advantage. Nothing else.
        You are a JUDGE, you are not part of the game.
        Provide the answer in the following JSON format:
        {'Game Summary':<game summary>,
        'Player 1 Status':{
            'Actions':<list of actions>,
            'Weaknesses':<list of weaknesses>,
            'Strengths':<list of strengths>,
            'Overall Status':<overall status>,
        }
        'Player 2 Status':{
            'Actions':<list of actions>,
            'Weaknesses':<list of weaknesses>,
            'Strengths':<list of strengths>,
            'Overall Status':<overall status>,
        },
        'Outcome':<Outcome So Far>,
        'Advantage':<Advantage>}
            }
        """
        judge_response = llmp_call(judge_prompt, judge_system_prompt, self.model,temperature=0,src='judge_call')
        return judge_response['message']['content']

    def random_event(self,dialogue):
        
        interactions = "\n".join(dialogue)
        random_events_system_prompt = 'You are a random events generator. Your tasks is to choose a random event that can happen that will affect the decisions. You are provided with a sequence of plays, you need to select a random event that can affect those plays. You are direct you only provide the needed text, no formalities, no greetings, nothing.'    
        random_event_prompt = interactions + '\n\n Considering this game, provide a random event that can affect the game and force the players to adapt. You must inform what is the effect of the random event on the players. Provide me only the event and effect on players. No unnecessary text! Provide the answer in markdown of the style **<event>**. \n**EFFECT ON PLAYER 1**: \n<effect_player_1>. **EFFECT ON PLAYER 2**: <effect_player_2>'
        judge_response = llmp_call(random_event_prompt, random_events_system_prompt, self.model,temperature=0.95, src='random_event_generator')
        return judge_response['message']['content']

    def sim_agent(self,user_prompt,iterations,rand_e_chance):
        
        moves = {}
        references_list = []
        dialogue = []
        comb_dialogue = []

        moves['opening_move'] = user_prompt
        iterations = int(iterations) # raise exception if input is invalid
        rand_e_chance = int(rand_e_chance)
        for i in range(iterations):
            print(f"\nTurn {i}")
            

            if i == 0:
                # Start the dialogue with opening
                dialogue.append(f"Player 2 did:: {moves['opening_move']}")
                comb_dialogue.append(f" ### Opening move:  \n {moves['opening_move']}\n ---")
                
                # Simulate generating move_adv_1_0 based on just the opening
                prompt = "\n".join(dialogue) + """\n You are Player 1. How will you counter it Player 2 latest move? Provide direct answer of steps to counter.\n Provide the response in the following JSON format:
                {Player:<your name>,
                'moves':{<move name>:<very direct move description>}
                }"""
                #print("Prompt to generate move_adv_1_0:\n", prompt)

                # ADV response
                adv_response = self.adv_agent.adv_agent_chat(prompt)
                # append response data ,i.e., counter move + references
                moves[f'move_adv_1_{i}'] = adv_response[0]
                references_list.append(adv_response[1])
                dialogue.append(f"Player 1 did: {moves[f'move_adv_1_{i}']}")
                comb_dialogue.append(f" ### Player 1 did:\n {moves[f'move_adv_1_{i}']}\n #### References:\n {adv_response[1]}\n ---")
                print(f'move_adv_1_{i} play')

            else:
                if np.random.random() < rand_e_chance:
                    print("Random Event")
                    _random_event = self.random_event(dialogue)
                    dialogue.append(f"\n**Random event**: {_random_event} \n")
                    comb_dialogue.append(f"\n**Random event**: {_random_event} \n")
                # Use the full dialogue to generate your next move
                prompt = "\n".join(dialogue) + """\n You are Player 2. How will you counter it Player 1 latest move? Provide direct answer of steps to counter.\n Provide the response in the following JSON format:
                {Player:<your name>,
                'moves':{<move name>:<very direct move description>}
                }"""
                #print(f"Prompt to generate move_adv_2_{i-1}:\n{prompt}")

                # CADV response
                adv_response = self.adv_agent.adv_agent_chat(prompt)
                moves[f'move_adv_2_{i-1}'] = adv_response[0]
                references_list.append(adv_response[1])
                dialogue.append(f"Player 2 did: {moves[f'move_adv_2_{i-1}']}")
                comb_dialogue.append(f"### Player 2 did:\n {moves[f'move_adv_2_{i-1}']}\n #### References:\n {adv_response[1]}\n ---")
                print(f'move_adv_2_{i-1} play')

                # Now generate adversary move based on updated dialogue
                prompt = "\n".join(dialogue) + """\n You are Player 1. How will you counter it Player 2 latest move? Provide direct answer of steps to counter.
                \n Provide the response in the following JSON format:
                {Player:<your name>,
                'moves':{<move name>:<very direct move description>}
                }"""
                #print(f"Prompt to generate move_adv_1_{i}:\n{prompt}")

                # ADV response
                adv_response = self.adv_agent.adv_agent_chat(prompt)
                moves[f'move_adv_1_{i}'] = adv_response[0]
                references_list.append(adv_response[1])
                dialogue.append(f"Player 1 did: {moves[f'move_adv_1_{i}']}")
                comb_dialogue.append(f"### Player 1 did:\n {moves[f'move_adv_1_{i}']}\n #### References:\n {adv_response[1]}\n ---")
                print(f'move_adv_1_{i} play')
                
        judge_eval = self.judge(moves)
        comb_dialogue.append(f"\n --- \n### Judge evaluation:\n {judge_eval}")    
        final_output = "\n".join(comb_dialogue)
        return final_output,dialogue,judge_eval # for testing
        #return final_output
        