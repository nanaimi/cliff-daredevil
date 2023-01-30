import Box2D as b2  # type: ignore
import numpy as np

CAR_WIDTH = 3.0
CAR_HEIGHT = 1.5


class CarModel(object):
    def __init__(self, world, init_x, init_y, init_velocity_x=0.0, init_velocity_y=0.0):
        self.world = world
        l, r, t, b = -CAR_WIDTH / 2, CAR_WIDTH / 2, CAR_HEIGHT, 0
        self.hull = world.CreateDynamicBody(
            position=(init_x, init_y),
            allowSleep=False,
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2PolygonShape(vertices=[(l, b), (l, t), (r, t), (r, b)]),
                    density=1.0,
                )
            ],
        )
        self.hull.linearVelocity = (init_velocity_x, init_velocity_y)
        self.hull.userData = self.hull
        radius = CAR_HEIGHT / 2.5
        wheels_positions = [
            (init_x + CAR_WIDTH / 4, init_y),
            (init_x - CAR_WIDTH / 4, init_y),
        ]
        wheel_drives = [True, False]
        self.wheels = []
        for position, drive in zip(wheels_positions, wheel_drives):
            wheel = self.world.CreateDynamicBody(
                position=position,
                fixtures=[
                    b2.b2FixtureDef(
                        shape=b2.b2CircleShape(radius=radius),
                        density=10.0,
                        friction=50.0,
                    )
                ],
            )
            rjd = b2.b2RevoluteJointDef(
                bodyA=self.hull,
                bodyB=wheel,
                anchor=wheel.worldCenter,
                motorSpeed=0.0,
                maxMotorTorque=100.0,
                enableMotor=drive,
            )
            wheel.linearVelocity = (init_velocity_x, init_velocity_y)
            wheel.angularVelocity = init_velocity_x / radius
            wheel.joint = self.world.CreateJoint(rjd)
            wheel.gas = 0.0
            wheel.brake = 0.0
            wheel.userData = wheel
            self.wheels.append(wheel)

    def brake(self, brake):
        for wheel in self.wheels:
            wheel.brake = brake

    def gas(self, gas):
        # Flip sign since wheels turn CCW.
        gas = -np.clip(gas, -1, 1)
        for wheel in self.wheels:
            diff = gas - wheel.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            wheel.gas += diff
            wheel.gas = gas

    def step(self):
        for wheel in self.wheels:
            wheel.joint.motorSpeed = 100.0 * (
                wheel.gas - wheel.brake * np.sign(wheel.angularVelocity)
            )

    def destroy(self):
        self.world.DestroyBody(self.hull)
        for wheel in self.wheels:
            self.world.DestroyBody(wheel)
        self.hull = None
