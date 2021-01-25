package wumpusWorld

import wumpusWorld.agent.NaiveAgent
import wumpusWorld.environment.{Environment, Percept}
import eu.timepit.refined.auto._

object WumpusWorld {

  def main(args: Array[String]) = {

    def runEpisode(env: Environment, agent: NaiveAgent, percept: Percept): Double = {
      val (nextAgent, nextAction) = agent.nextAction(percept)
      println(s"Action: $nextAction")
      val (nextEnvironment: Environment, nextPercept: Percept) = env.applyAction(nextAction)
      println(nextEnvironment.visualize)
      println(nextPercept.show)
      nextPercept.reward + (if (!nextPercept.isTerminated) runEpisode(nextEnvironment, agent, nextPercept) else 0.0)
    }

    val (initialEnv: Environment, initialPercept: Percept) = Environment(4, 4, 0.2, false)

    val agent = new NaiveAgent
    val totalReward: Double = runEpisode(initialEnv, agent, initialPercept)
    println(s"Total reward: $totalReward")
  }
}
