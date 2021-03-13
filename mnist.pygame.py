import pygame
import cv2
import tensorflow as tf

def main():

    # loading trained model
    model = tf.keras.models.load_model("classifier")

    # pygame interface
    pygame.init()

    pygame.display.set_mode((280,280))
    pygame.display.set_caption("Modelling MNIST")
    screen = pygame.display.get_surface()
    game = Game(screen, model)
    game.play()

    pygame.quit()

class Game:

    def __init__(self, screen, model):

        # classifier variables
        self.model = model
        self.image = None
        self.pred = None

        # pygame variables
        self.screen = screen
        self.predScreen = None
        self.FPS = 120
        self.game_Clock = pygame.time.Clock()
        self.close_clicked = False

        # brush variables
        self.dotRad = 10
        self.brushColor = pygame.Color("white")
        self.drag = False
        self.cursorPos = None

    def play(self):
        """main game loop"""
        while not self.close_clicked:
            self.handle_events()
            self.draw()
            self.update()
            self.game_Clock.tick(self.FPS)

    def handle_events(self):
        """mouse and keyboard event handler"""
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.close_clicked = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.drag = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.drag = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    img = self.screenShot()
                    self.image = self.convertImage(img, 28)  # ready for prediction
                if event.key == pygame.K_LSHIFT:
                    self.brushColor = pygame.Color("black")
                if event.key == pygame.K_RSHIFT:
                    self.brushColor = pygame.Color("white")
                if event.key == pygame.K_p:
                    self.pred = self.makePred(self.model, self.image)  # predicting
                    # print(self.pred)
                    self.showPred(self.pred)

    def draw(self):
        """draws the brush strokes onto the screen"""
        if self.drag:
            pygame.draw.circle(self.screen, self.brushColor, self.cursorPos, self.dotRad)
            # rect = pygame.Rect(self.cursorPos[0]-self.dotRad/2, self.cursorPos[1]-self.dotRad/2, self.dotRad, self.dotRad)
            # pygame.draw.rect(self.screen, self.brushColor, rect)

    def update(self):
        """updates the position of the cursor and the display with brush strokes"""
        self.cursorPos = pygame.mouse.get_pos()
        pygame.display.update()

    def screenShot(self):
        """saves the screen as "screenshot.jpg" in cwd of this .py file"""
        pygame.image.save(self.screen, "screenshot.jpg")

    def convertImage(self, img, imgSize):
        """
        converts drawn image into appropriate input data type for the trained model
        inputs: screenshot image and the inpur image size
        returns: correctly formatted image for model.predict()
        """
        img = cv2.imread("screenshot.jpg", 0)
        img = tf.expand_dims(img, -1)  # from 28x28 to 28x28x1
        img = tf.divide(img, 255)  # normalize
        img = tf.image.resize(img, [imgSize, imgSize])   # resize acc to the input
        img = tf.reshape(img, [1, imgSize, imgSize, 1])  # reshape to add batch dimension

        return img

    def makePred(self, model, image):
        """
        makes the prediction on image
        inputs: loaded tf.keras model, converted input image
        returns: prediction based on the highest probability
        """
        pred = model.predict(image)
        pred = tf.argmax(pred, axis=-1).numpy()

        return pred

    def showPred(self, pred):
        """
        displays the self.pred on screen
        """
        pred = str(pred[0])
        textPos = (5, 2)
        textColor = pygame.Color("white")
        textFont = pygame.font.SysFont('freesansbold.ttf', 64)
        textImage = textFont.render(pred, True, textColor)
        self.screen.blit(textImage, textPos)


main()

