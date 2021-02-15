class AgentState:
    def __init__(self, location, orientation, has_gold, has_arrow, is_alive):
        self.location = location
        self.orientation = orientation
        self.has_gold = has_gold
        self.has_arrow = has_arrow
        self.is_alive = is_alive

    def __copy__(self):
        return AgentState(self.location,
                          self.orientation,
                          self.has_gold,
                          self.has_arrow,
                          self.is_alive)

    def turn_left(self):
        _new_agent_state = AgentState.__copy__(self)
        _new_agent_state.orientation = self.orientation.turn_left()
        return _new_agent_state

    def turn_right(self):
        _new_agent_state = AgentState.__copy__(self)
        _new_agent_state.orientation = self.orientation.turn_right()
        return _new_agent_state

    def forward(self, grid_width, grid_height):
