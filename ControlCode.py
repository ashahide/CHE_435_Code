class SystemAnalysis:

    def RootLocus(self):

        import control.matlab
        import matplotlib.pyplot as plt 

        if self.G_CL and self.G_OL.G:
            import matplotlib.pyplot as plt 

            plt.figure(figsize = (14,14)) 
            plt.subplot(2,1,1)
            control.matlab.rlocus(self.G_OL.G, label = 'OL')

            plt.legend()

            plt.subplot(2,1,2)
            control.matlab.rlocus(self.G_CL, label = 'OL')

            plt.legend()

        plt.show()

        return

    def BodePlot(self, Units = None):
        import control

        try:
            Sys = self.G_CL 
        except:
            Sys = self.G

        if Units == 'dB':
            BodeMagnitude, BodePhase, BodeOmega = control.bode_plot(Sys, dB = True)
        elif Units == 'Hz':
            BodeMagnitude, BodePhase, BodeOmega = control.bode_plot(Sys, Hz = True)
        elif Units == 'degree':
            BodeMagnitude, BodePhase, BodeOmega = control.bode_plot(Sys, deg = True)
        else:
            BodeMagnitude, BodePhase, BodeOmega = control.bode_plot(Sys)

        return BodeMagnitude, BodePhase, BodeOmega   

class TransferFunction(SystemAnalysis):

    """
    Links to cite:
    [1] Python control toolbox from CalTech
    [2] http://techteach.no/python_control/python_control.pdf
    """

    def __init__(self, Numerator, Denominator, TFinal = 10, NumPoints = 1000, TimeDelay = None, Systemlabel = None):

        import numpy as np 

        self.Numerator   = Numerator
        self.Denominator = Denominator
        self.TimeDelay   = TimeDelay

        self.t           = np.linspace(0, TFinal, NumPoints)

        self.SystemLabel = Systemlabel

        self.SysType     = 'Open Loop'

        """
        This is just to supress the output
        https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
        """

        from contextlib import contextmanager
        import sys, os

        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:  
                    yield
                finally:
                    sys.stdout = old_stdout

        """
        Using method for time delay creation from [2]
        """
        import control 

        if self.TimeDelay is not None:

            PadeNumerator, PadeDenominator = control.pade(self.TimeDelay, 1)

            G_Pade = control.tf(PadeNumerator, PadeDenominator)

            G_NoDelay = control.tf(self.Numerator, self.Denominator)

            G = G_Pade*G_NoDelay
            

        else:
            G = control.tf(self.Numerator, self.Denominator)

        self.G = G

        """
        Define poles, zeros, and damping (if 2nd order)
        """

        import control.matlab
        import numpy as np

        self.Zeros = control.matlab.zero(self.G)

        with suppress_stdout():
            if len(Denominator) > 2:
                self.PoleFrequencies, self.Damping, self.Poles = control.matlab.damp(self.G)
            else:
                self.PoleFrequencies, _, self.Poles = control.matlab.damp(self.G)
                self.Damping = None 

        if self.Damping is not None and self.Damping[0] < 1:
            self.DampingType = 'Underdamped'
        elif self.Damping is not None and self.Damping[0] > 1:
            self.DampingType = 'Overdamped'
        else:
            self.DampingType = 'Critically Damped'

        """
        Print Characteristics
        """
        print("\n######################################################")
        if self.SystemLabel:
            print(f"##### {self.SystemLabel} Characteristics")
        else:
            print("##### Transfer Function Characteristics")
        print("######################################################")
        if self.Zeros:
            print(f"## Zeros: {self.Zeros}")
        else:
            print(f"## Zeros: None")
        if self.Damping is not None:
            try:
                print(f"## Damping Coefficient: {np.round(self.Damping[0],3)} ({self.DampingType})")
            except:
                print(f"## Damping Coefficient: {np.round(self.Damping,3)} ({self.DampingType})")
        print(f"## Poles:               {[np.round(i,3) for i in self.Poles]}")
        print("######################################################")

    def InputFunction(self, Magnitude, Type, InputEndTime = None):

        import control.matlab
        import control
        import numpy as np 
        


        if Type == 'Step':
            Output, Time = control.matlab.step(self.G, self.t)
            self.Time              = Time

        elif Type == 'Impulse':
            Output, Time = control.matlab.impulse(self.G, self.t)
            self.Time              = Time

        elif Type == 'Square':

            # U with original Time
            self.U_Time = np.zeros_like(self.t)
            self.U_Time[0:InputEndTime] = 1

            # U Corrected for samples
            U = np.zeros_like(self.t)
            U[0:InputEndTime*int(len(self.t)/self.t[-1])] = 1

            Output, Time, _  = control.matlab.lsim(self.G, U=U, T=self.t)
            self.Time        = Time
            self.U           = U*Magnitude

        self.InputEndTime = InputEndTime
        self.Type         = Type
        self.Magnitude    = Magnitude
        self.Input        = Type
        self.Output       = Magnitude*Output


        return 
        # return Magnitude*Output, self.Time

def PlotResponse(System):
    import matplotlib.pyplot as plt 
    plt.style.use('seaborn-darkgrid')

    if System.Input == 'Square':
        color = 'g'
    
    elif System.Input == 'Impulse':
        color = 'r'

    else:
        color = 'b'

    LineWidth = 3

    fig, ax = plt.subplots()

    ax.plot(System.t, System.Output, color = color, linewidth = LineWidth, label = "TF")
    ax.set_ylabel("Output", color = color, fontsize = 14)
    ax.tick_params(axis = 'y', labelcolor = color, labelsize = 12)
    ax.set_xlabel('Time', fontsize = 14)

    ax2 = ax.twinx()
    ax2.set_ylabel("Input", color = 'k', fontsize = 14)
    ax2.tick_params(axis = 'y', labelcolor = 'k', labelsize = 12)


    if System.Input == 'Square':
        ax2.hlines(System.Magnitude, 0, System.InputEndTime, color = 'k',  linewidth = LineWidth, label = f'Square Input, Magnitude = {System.Magnitude}')
        ax2.vlines(0, 0, System.Magnitude, color = 'k',  linewidth = LineWidth)
        ax2.vlines(System.InputEndTime, 0, System.Magnitude, color = 'k',  linewidth = LineWidth)

    elif System.Input == 'Impulse':
        ax2.vlines(0, 0, System.Magnitude, linewidth = LineWidth, color = 'k', label = f'Impulse Input, Magnitude = {System.Magnitude}')

    else:
        ax2.hlines(System.Magnitude, 0, System.t[-1], color = 'k', linewidth = LineWidth, label = f'{System.Input} Input, Magnitude = {System.Magnitude}')
        try:
            ax2.vlines(System.t[0], 0, System.Magnitude, color = 'k',  linewidth = LineWidth)
        except:
            ax2.vlines(0, 0, System.Magnitude, color = 'k',  linewidth = LineWidth)

    plt.show()

class DesignFeedback(SystemAnalysis):

    def __init__(self, OpenLoopSystem, SysLabel = None):
        self.G_OL               = OpenLoopSystem
        self.SystemLabel        = SysLabel

        self.SysType     = 'Closed Loop'

    def DesignP(self, Kp = None):
        import control.matlab as cl 
        import control 
        import copy

        Sys1 = self.G_OL.G
        self.G_Controller = control.tf([0, Kp], [0, 1])

        self.G_CL = cl.feedback(Sys1, self.G_Controller)  

        DesignFeedback.SimulateCLSystem(self)

        return copy.deepcopy(self)

    def DesignPI(self, Kp = None, Ki = None):
        import control.matlab as cl 
        import control 
        import copy

        Sys1 = self.G_OL.G
        self.G_Controller = control.tf([Kp*Ki, Kp], [Ki, 0])

        self.G_CL = cl.feedback(Sys1, self.G_Controller)

        DesignFeedback.SimulateCLSystem(self)

        return copy.deepcopy(self)

    def DesignPID(self, Kp = None, Ki = None, Kd = None):
        import control.matlab as cl 
        import control 
        import copy

        Sys1 = self.G_OL.G
        self.G_Controller = control.tf([Kd*Ki*Kp, Kp*Ki, Kp], [Ki, 0])

        self.G_CL = cl.feedback(Sys1, self.G_Controller)

        DesignFeedback.SimulateCLSystem(self)

        return copy.deepcopy(self)

    def SimulateCLSystem(self):
        import control.matlab
        import control
        import numpy as np 
        

        t = np.linspace(0, self.G_OL.TFinal, self.G_OL.NumPoints)

        if self.G_OL.Type == 'Step':
            Output, Time = control.matlab.step(self.G_CL, t)
            self.Time              = Time

        elif self.G_OL.Type == 'Impulse':
            Output, Time = control.matlab.impulse(self.G_CL, t)
            self.Time              = Time

        elif self.G_OL.Type == 'Square':
            self.T = np.linspace(0, self.G_OL.TFinal, self.G_OL.NumPoints)

            # U with original Time
            self.U_Time = np.zeros_like(self.G_OL.T)
            self.U_Time[0:self.G_OL.InputEndTime] = self.G_OL.Magnitude

            # U Corrected for samples
            U = np.zeros_like(self.G_OL.T)
            U[0:self.G_OL.InputEndTime*int(self.G_OL.NumPoints/self.G_OL.TFinal)] = 1

            Output, Time, _  = control.matlab.lsim(self.G_CL, U=U, T=self.T)
            self.Time        = Time
            self.U           = U

        self.Magnitude  = self.G_OL.Magnitude
        self.Input      = self.G_OL.Type
        self.Output     = self.G_OL.Magnitude*Output

def CompareResults(*Systems, YUnit = None):

    if any([i.SysType == 'Closed Loop' for i in Systems]):

        import matplotlib.pyplot as plt
        plt.style.use('seaborn-darkgrid')

        LineWidth = 3

        """
        figure 1
        """

        plt.figure(figsize=(14,12))

        plt.tight_layout()

        for i in range(len(Systems)):
            if Systems[i].SystemLabel is not None:
                plt.plot(Systems[i].Output, linewidth = LineWidth, label = f'{Systems[i].SystemLabel}')
            else:
                plt.plot(Systems[i].Output, linewidth = LineWidth, label = f'TF {i+1}')

        plt.legend() 

        plt.show()


    else:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-darkgrid')

        LineWidth = 3

        """
        Figure 1: Plot individual systems wit inputs
        """
        plt.figure(figsize=(14,12))

        plt.tight_layout()

        for i in range(len(Systems)):

            plt.subplot(3,2,i+1)

            if Systems[i].Input == 'Square':
                plt.plot(Systems[i].Time, Systems[i].U, color = 'k',  linewidth = LineWidth, label = f'Square Input, Magnitude = {Systems[i].Magnitude}')
                plt.vlines(0, 0, Systems[i].Magnitude, linewidth = LineWidth, color = 'k')

                color = 'g'


            elif Systems[i].Input == 'Impulse':
                plt.vlines(0, 0, Systems[i].Magnitude, color = 'k', linewidth = LineWidth, label = 'Impulse Input, Magnitude = {}'.format(Systems[i].Magnitude))

                color = 'r'

            else:
                plt.hlines(Systems[i].Magnitude, 0, Systems[i].Time[-1], color = 'k', linewidth = LineWidth, label = '{} Input, Magnitude = {}'.format(Systems[i].Input, Systems[i].Magnitude))
                try:
                    plt.vlines(Systems[i].T[0], 0, Systems[i].Magnitude, color = 'k',  linewidth = LineWidth)
                except:
                    plt.vlines(0, 0, Systems[i].Magnitude, color = 'k',  linewidth = LineWidth)

                color = 'b'
            if Systems[i].SystemLabel:
                plt.plot(Systems[i].Time, Systems[i].Output, color = color, linewidth = LineWidth, label = f'{Systems[i].SystemLabel} (Time Delay = {Systems[i].TimeDelay})')
            else:
                plt.plot(Systems[i].Time, Systems[i].Output, color = color, linewidth = LineWidth, label = 'TF {} (Time Delay = {})'.format(i+1, Systems[i].TimeDelay))

            plt.xlabel('Time')
            plt.ylabel('Output')
            plt.legend(loc = 'best')

        """
        Figure 2: Plot all systems at once
        """

        def SelectColor(Type):
            if Type == 'Square':
                color = 'g'

            elif Type == 'Impulse':
                color = 'r'

            elif Type == 'Step':
                color = 'b'

            return color

        StyleList = ['-', '--', '-o', '-x', '-+', '-*']
        ColorList = []

        plt.figure(figsize=(14,12))

        for i in range(len(Systems)):
            ColorList.append(SelectColor(Systems[i].Input))

            style = StyleList[ColorList.count(SelectColor(Systems[i].Input)) - 1]

            if Systems[i].SystemLabel:
                plt.plot(Systems[i].Time, Systems[i].Output, SelectColor(Systems[i].Input) + style, linewidth = LineWidth, \
                    label =  f'{Systems[i].SystemLabel} ({Systems[i].Input} Input, Magnitude = {Systems[i].Magnitude}, Time Delay = {Systems[i].TimeDelay})')

            else:
                plt.plot(Systems[i].Time, Systems[i].Output, SelectColor(Systems[i].Input) + style, linewidth = LineWidth, \
                    label =  f'TF {i+1} ({Systems[i].Input} Input, Magnitude = {Systems[i].Magnitude}, Time Delay = {Systems[i].TimeDelay})')

        
            plt.legend()
            plt.xlabel('Time', fontsize = 14)
            plt.ylabel('Output', fontsize = 14)

        plt.show()

