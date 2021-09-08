
#input format: scattering cross section magnitude (R), emiss cross section ((x,y) pointlist)

granularity = 20
new_pointlist = []

for point in pointlist
    for slice in [1:granularity]
        theta = slice*pi/granularity
        point_x = cos(theta)*point[1]
        point_y = point[2]
        point_z = sin(theta)*point[1]
        push!(new_pointlist, [point_x, point_y, point_z])
    end
end

