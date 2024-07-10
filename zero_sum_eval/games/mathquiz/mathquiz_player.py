# I took inspiration from https://github.com/carlini/chess-llm and https://github.com/mlabonne/chessllm
# Shout out to the maintainers and authors of these repositories!

from zero_sum_eval.player import Player
import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from dspy.teleprompt import LabeledFewShot, BootstrapFewShot, MIPROv2, BootstrapFewShotWithRandomSearch
import functools, json
from random import shuffle


class GenerateQuestion(dspy.Signature):
    """Given a target answer, generate a challenging math question that has the target answer"""
    
    target_answer = dspy.InputField(desc="target answer for the question")
    math_question = dspy.OutputField(desc="math question with the target answer")

class AnswerQuestion(dspy.Signature):
    """Given a math question, answer the question"""
    
    math_question = dspy.InputField(desc="math question to answer")
    answer = dspy.OutputField(desc="answer to the math question")

class GenerateQuestionCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot_question = dspy.ChainOfThought(GenerateQuestion)


    def forward(self, target_answer):
        cot_out = self.cot_question(target_answer=target_answer)
        return cot_out

class AnswerQuestionCoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot_answer = dspy.ChainOfThought(AnswerQuestion)

    def forward(self, math_question):
        cot_out = self.cot_answer(math_question=math_question)
        return cot_out

class MathQuizTeacher(Player):
    def __init__(self, llm_model, max_tries=4, **kwargs):
        super().__init__(**kwargs)
        self.llm_model = llm_model
        self.max_tries = max_tries
        self.question_module = GenerateQuestionCoT()
        self.answer_module = AnswerQuestionCoT()
        # self.optimized_module = self.optimize_prompts() if self.optimize else None
        dspy.configure(trace=[])

    def make_move(self, game_state):
        """
        Abstract method for making a move based on the current game state.
        
        Parameters:
        game_state (GameState): The current state of the game
        
        Returns:
        str: The move made by the player
        """
        export = game_state.export()
        current_role = game_state.roles[0]
        with dspy.context(lm=self.llm_model):
            if current_role == "TeacherGenerateQuestion":
                trace = self.question_module(export['target'])
                return trace.math_question
            elif current_role == "TeacherAnswerQuestion":
                trace = self.answer_module(export['environment'])
                return trace.answer
    
class MathQuizQuestion(Player):
    def __init__(self, llm_model, max_tries=4, **kwargs):
        super().__init__(**kwargs)
        self.llm_model = llm_model
        self.max_tries = max_tries
        self.answer_module = AnswerQuestionCoT()
        # self.optimized_module = self.optimize_prompts() if self.optimize else None
        dspy.configure(trace=[])

    def make_move(self, game_state):
        """
        Abstract method for making a move based on the current game state.
        
        Parameters:
        game_state (GameState): The current state of the game
        
        Returns:
        str: The move made by the player
        """
        export = game_state.export()
        with dspy.context(lm=self.llm_model):
            trace = self.answer_module(export['environment'])
            return trace.answer

        
    
