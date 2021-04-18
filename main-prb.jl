using Optim
using DelimitedFiles
using SymEngine

c0 = 6.832
c1 = 4.064
c2 = -0.374
c3 = -0.095
K = 69518.0
G = 47352.0
pn = 4
grid = 20
z = pi / 60
a = 2.47
ay = a*sqrt(3)/2
A = [0.5 sqrt(3)/2 ; sqrt(3)/2 -0.5]

struct vector
    x
    y
end

function Ur(bx, by, ui)
    #Ux(px,py)=Ui[pn*py+px+1]
    #Uy(px,py)=Ui[pn*pn+pn*py+px+1]
    x = 0
    y = 0
    for py in 0:pn-1
        for px in 0:pn-1
            e = cos((bx*px/a + by*py/ay)*2*pi)
            x += ui[pn*py+px+1] * e
            y += ui[pn*pn+pn*py+px+1] * e
		end
	end
    ret = vector(x,y)
    return ret
end

function Einter(bx, by, ui)
    ux = 0
    uy = 0
	if bx != 0 && by/bx > 1/sqrt(3) || bx == 0 && by != 0
		bxs,bys = A*[bx;by]
		for py in 0:pn-1
			for px in 0:pn-1
				e = cos((bxs*px/a + bys*py/ay)*2*pi)
				ux += ui[pn*py+px+1] * e
				uy += ui[pn*pn+pn*py+px+1] * e
			end
		end
		ux,uy = A*[ux;uy]
	else	
		for py in 0:pn-1
			for px in 0:pn-1
				e = cos((bx*px/a + by*py/ay)*2*pi)
				ux += ui[pn*py+px+1] * e
				uy += ui[pn*pn+pn*py+px+1] * e
			end
		end
	end	
    bx = bx + 2*ux
    by = by + 2*uy
    v = 2pi/a * (bx - 1/sqrt(3) * by)
    w = 2pi/a * 2/sqrt(3) * by
    ret = c0
    ret += c1 * (cos(v) + cos(w) + cos(v+w))
    ret += c2 * (cos(v+2*w) + cos(v-w) + cos(2*v+w))
    ret += c3 * (cos(2*v) + cos(2*w) + cos(2*v+2*w))
    return ret
end

function Eintra(bx, by, ui)
    ret = 0
    xx = 0
    xy = 0
    yx = 0
    yy = 0
	if bx != 0 && by/bx > 1/sqrt(3) || bx == 0 && by != 0
		bx,by = A*[bx;by]
	end
	
	for py in 0:pn-1
		for px in 0:pn-1
			e = -sin((bx*px/a + by*py/ay)*2*pi)
			xx += ui[pn*py+px+1] * e * px*2*pi/a
			yx += ui[pn*pn+pn*py+px+1] * e * px*2*pi/a
			xy += ui[pn*py+px+1] * e * py*2*pi/ay
			yy += ui[pn*pn+pn*py+px+1] * e * py*2*pi/ay
		end
	end
	
	if bx != 0 && by/bx > 1/sqrt(3) || bx == 0 && by != 0
		xx,yx,xy,yy = [xx:yx xy:yy]*A
	end
	#ret = G*(Uxx+Uyy)^2 + K * ((Uxx-Uyy)^2+(Uxy+Uyx)^2)
	#bx = -zry; by = zrx; Uxx = zuxy; Uxy = -zuxx; Uyx = zuyy; Uyy = -zuyx          
    ret = G*(xy-yx)^2 + K * ((xy+yx)^2 + (-xx+yy)^2)
    return ret
end
			
function Etotal(ui)
    ret = 0
    for by in 0 : ay/grid : ay
        for bx in by/sqrt(3) : a/grid : a+by/sqrt(3)
            ret += Eintra(bx, by, ui)
            ret += Einter(bx, by, ui)
		end
	end
    return ret * a*ay/grid^2
end

function g!(Gra, ui)
	for i in 1:2pn^2
		y = e[i]
		for t in 1:2pn^2
			y = subs(y, s[t]=>ui[t])
		end
		Gra[i] = y
	end
end
	
function h!(H, ui)
	for i in 1:2pn^2
		for j in 1:2pn^2
			y = ee[i,j]
			for t in 1:2pn^2
			y = subs(y, s[t]=>ui[t])
			end
			H[i,j] = y
		end
	end
end	

s = [symbols("s[$i]") for i in 1:2pn^2]
f = Etotal(s)
e = Array{Union{Missing, Basic}}(missing,2pn^2)
for i in 1:2pn^2
	e[i] = diff(f,s[i])
end
ee = Array{Union{Missing, Basic}}(missing,2pn^2,2pn^2)
for i in 1:2pn^2
	for j in 1:2pn^2
		ee[i,j] = diff(f,s[i],s[j])
	end
end
	

function test()
    println(Etotal(zeros(2pn^2)))
    tt = ['x' 'y' 'u' 'u']
    res = optimize(Etotal, g!, h!, zeros(2pn^2)*1e-4, method = LBFGS(), iterations= 10000)
    println(res)
	ans = Optim.minimizer(res)
    for by in 0 : ay/grid : ay
        for bx in by/sqrt(3) : a/grid : a+by/sqrt(3)
            ur = Ur(bx, by, ans)
            tt = [tt;[bx by ur.x ur.y]]
		end
	end
    writedlm("soat.txt", tt, '\t')
end
    
test()

